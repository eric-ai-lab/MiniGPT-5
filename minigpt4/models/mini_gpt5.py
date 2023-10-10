import torch.nn as nn
import torch

from minigpt4.common.registry import registry
from minigpt4.models.mini_gpt4 import MiniGPT4
from transformers import StoppingCriteria, StoppingCriteriaList
from constants import *

from peft import (
    LoraConfig,
    PeftType,
    get_peft_model,
    PrefixTuningConfig,
    PromptLearningConfig,
    PrefixEncoder,
    TaskType
)

import types
from .generation_helper import sample

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False
    
@registry.register_model("minigpt5")
class MiniGPT5(MiniGPT4):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to(PRECISION)
        all_img_tokens = ALL_IMG_TOKENS + ["<ImageHere>"]
        self.num_new_tokens = self.llama_tokenizer.add_special_tokens(
            {   
                # "pad_token": DEFAULT_PAD_TOKEN,
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                # "unk_token": DEFAULT_UNK_TOKEN,
                "additional_special_tokens": all_img_tokens
            }
        )
        if self.num_new_tokens > 0:
            self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
            input_embeddings = self.llama_model.get_input_embeddings().weight.data
            output_embeddings = self.llama_model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-self.num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-self.num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-self.num_new_tokens:] = input_embeddings_avg
            output_embeddings[-self.num_new_tokens:] = output_embeddings_avg
        
        if len(ALL_IMG_TOKENS):
            self.output_img_id = self.llama_tokenizer.convert_tokens_to_ids(ALL_IMG_TOKENS[0])
        self.input_img_id = self.llama_tokenizer.convert_tokens_to_ids('<ImageHere>')

        self.llama_model.get_input_embeddings().to(TRAINABLE_PRECISION)
        self.llama_model.get_output_embeddings().to(TRAINABLE_PRECISION)

        # original_input_embeddings = self.llama_model.get_input_embeddings().weight.data.clone()
        # original_output_embeddings = self.llama_model.get_output_embeddings().weight.data.clone()
        # self.register_buffer("original_input_embeddings", original_input_embeddings, persistent=False)
        # self.register_buffer("original_output_embeddings", original_output_embeddings, persistent=False)
        input_embed_grad_mask = torch.ones_like(self.llama_model.get_input_embeddings().weight.data)
        output_embed_grad_mask = torch.ones_like(self.llama_model.get_output_embeddings().weight.data)
        input_embed_grad_mask[:-self.num_new_tokens] = 0
        output_embed_grad_mask[:-self.num_new_tokens] = 0
        self.register_buffer("input_embed_grad_mask", input_embed_grad_mask, persistent=False)
        self.register_buffer("output_embed_grad_mask", output_embed_grad_mask, persistent=False)

        self.base_model_prepare_inputs_for_generation = self.llama_model.prepare_inputs_for_generation
        self.base_model_torch_dtype = self.llama_model.dtype
        self.llama_model.sample = types.MethodType(sample, self.llama_model)
        self.llama_model.output_img_id = self.output_img_id

        self.llama_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation     

        self.using_lora = USE_LORA
        self.using_prefix_tuning = USE_PREFIX_TUNING
        if self.using_lora:
            print("Using LoRA")
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            lora_r = 8
            lora_alpha = 16
            lora_dropout = 0.05
            self.lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                modules_to_save=['lm_head','embed_tokens']
            )
            self.llama_model = get_peft_model(self.llama_model, self.lora_config)
            self.llama_model.base_model.model.model.embed_tokens.original_module.weight.requires_grad = False
            self.llama_model.base_model.model.lm_head.original_module.weight.requires_grad = False
        else:
            for name, param in self.llama_model.named_parameters():
                if "lm_head" in name or "embed_tokens" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        llama_config = self.llama_model.config
        if self.using_prefix_tuning:
            num_virtual_tokens = 5
            self.peft_config = PrefixTuningConfig(peft_type="P_TUNING",
                                              task_type=TaskType.CAUSAL_LM, 
                                              num_virtual_tokens=num_virtual_tokens,
                                              num_layers=llama_config.num_hidden_layers,
                                              num_attention_heads = llama_config.num_attention_heads,
                                              token_dim=llama_config.hidden_size,
                                              encoder_hidden_size=1024,
                                              prefix_projection=True)
            self.prefix_encoder = PrefixEncoder(self.peft_config).to(TRAINABLE_PRECISION)
            self.prefix_tokens = torch.arange(num_virtual_tokens).long()

        for name, param in self.llama_proj.named_parameters():
            param.requires_grad = False

    def get_prompt(self, batch_size, device, inference_mode=False):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.peft_config.num_virtual_tokens,
            self.peft_config.num_layers * 2,
            self.peft_config.num_attention_heads,
            self.peft_config.token_dim//self.peft_config.num_attention_heads
                )
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values
    
    def encode_img(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")
        with torch.no_grad():
            with torch.autocast('cuda'):
                image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

                inputs_llama = self.llama_proj(query_output.last_hidden_state)
                atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama
    
    def input_warp(self, input_ids, attention_mask, labels=None, input_image=None, input_image_feature=None):
        assert input_ids.shape[0] == 1, "wrapping each sample individually"

        bos = torch.ones([1, 1],
                        dtype=input_ids.dtype,
                        device=input_ids.device) * self.llama_tokenizer.bos_token_id
        if self.using_lora:
            bos_embeds = self.llama_model.base_model.model.model.embed_tokens(bos)
        else:
            bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = torch.ones([1, 1],dtype=attention_mask.dtype,device=attention_mask.device)
        if labels is not None:
            labels_bos = torch.ones([1, 1],dtype=labels.dtype,device=labels.device) * -100
            wrapped_labels = labels_bos
        else:
            wrapped_labels = None
        
        wrapped_img_embeds, wrapped_atts_img = bos_embeds, atts_bos
        input_img_idx = (input_ids == self.input_img_id).nonzero(as_tuple=True)
        start_idx = 0
        if len(input_img_idx[0]) > 0:
            assert input_image is not None or input_image_feature is not None, 'input_image or input_image_feature should be provided'

            if input_image_feature is not None:
                img_embeds = input_image_feature
                atts_img = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to(img_embeds.device)
            else:
                img_embeds, atts_img = self.encode_img(input_image)
            
            if labels is not None:
                img_label = torch.ones_like(atts_img, dtype=torch.long).to(img_embeds.device) * -100

            for i in range(len(input_img_idx[1])):
                p_before = input_ids[:, start_idx:input_img_idx[1][i]]
                p_before_attention_mask = attention_mask[:, start_idx:input_img_idx[1][i]]
                p_before_embeds = self.get_input_embeddings(p_before)
                wrapped_img_embeds = torch.cat([wrapped_img_embeds, p_before_embeds, img_embeds[i:i+1]], dim=1)
                wrapped_atts_img = torch.cat([wrapped_atts_img, p_before_attention_mask, atts_img[i:i+1]], dim=1)
                if labels is not None:
                    p_before_labels = labels[:, start_idx:input_img_idx[1][i]]
                    wrapped_labels = torch.cat([wrapped_labels, p_before_labels, img_label[i:i+1]], dim=1)
                start_idx = input_img_idx[1][i] + 1
            
        p_before = input_ids[:, start_idx:]
        p_before_attention_mask = attention_mask[:, start_idx:]
        p_before_embeds = self.get_input_embeddings(p_before)
        wrapped_img_embeds = torch.cat([wrapped_img_embeds, p_before_embeds], dim=1)
        wrapped_atts_img = torch.cat([wrapped_atts_img, p_before_attention_mask], dim=1)
        if labels is not None:
            p_before_labels = labels[:, start_idx:]
            wrapped_labels = torch.cat([wrapped_labels, p_before_labels], dim=1)
        return wrapped_img_embeds, wrapped_atts_img, wrapped_labels
            
    def forward(self, input_ids, labels, attention_mask, input_images=None, input_img_features=None, output_hidden_states=True):
        batch_size = input_ids.shape[0]
        all_input_embeds, all_attention, all_labels = [], [], []
        for b in range(batch_size):
            if input_img_features is not None:
                wrapped_img_embeds, wrapped_atts_img, wrapped_labels = self.input_warp(input_ids[b:b+1], attention_mask[b:b+1], labels[b:b+1], input_image_feature=input_img_features[b])
            elif input_images is not None:
                wrapped_img_embeds, wrapped_atts_img, wrapped_labels = self.input_warp(input_ids[b:b+1], attention_mask[b:b+1], labels[b:b+1], input_images[b])

            all_input_embeds.append(wrapped_img_embeds)
            all_attention.append(wrapped_atts_img)
            all_labels.append(wrapped_labels)

        #add padding features for batch
        max_len = max([x.shape[1] for x in all_input_embeds])
        for i in range(len(all_input_embeds)):
            if all_input_embeds[i].shape[1] < max_len:
                pad_len = max_len - all_input_embeds[i].shape[1]
                pad_embeds = torch.zeros([all_input_embeds[i].shape[0], pad_len, all_input_embeds[i].shape[2]]).to(all_input_embeds[i].device)
                pad_atts = torch.zeros([all_attention[i].shape[0], pad_len]).to(all_attention[i].device)
                pad_labels = torch.ones([all_labels[i].shape[0], pad_len], dtype=torch.long).to(all_labels[i].device) * -100
                all_input_embeds[i] = torch.cat([all_input_embeds[i], pad_embeds], dim=1)
                all_attention[i] = torch.cat([all_attention[i], pad_atts], dim=1)
                all_labels[i] = torch.cat([all_labels[i], pad_labels], dim=1)
        
        all_input_embeds = torch.cat(all_input_embeds, dim=0)
        all_attention = torch.cat(all_attention, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        past_key_values = None
        if self.using_prefix_tuning:
            device = all_input_embeds.device
            past_key_values = self.get_prompt(batch_size=batch_size, device=device)
            prefix_attention_mask = torch.ones(batch_size, self.peft_config.num_virtual_tokens).to(device)
            all_attention = torch.cat([prefix_attention_mask, all_attention], dim=1)
            # prefix_labels = torch.ones(batch_size, self.peft_config.num_virtual_tokens, dtype=wrapped_labels.dtype).to(device) * -100
            # wrapped_labels = torch.cat([prefix_labels, wrapped_labels], dim=1)

        outputs = self.llama_model(
                inputs_embeds=all_input_embeds,
                attention_mask=all_attention,
                return_dict=True,
                labels=all_labels,
                output_hidden_states=output_hidden_states,
                past_key_values=past_key_values,
            )
        output_token_index = (all_labels == self.output_img_id).nonzero()

        if len(output_token_index):
            addon_index = torch.ones_like(output_token_index)*(-1)
            addon_index[:, 0] = 0
            output_token_index += addon_index
        
        return outputs, output_token_index
    
    def predict(self, instruction, input_image, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, task_name=None, output_hidden_states=True, force_generation=False):

        sample_inputs = self.llama_tokenizer(instruction, return_tensors="pt", add_special_tokens = False).to(self.device)
        input_ids = sample_inputs.input_ids
        attention_mask = sample_inputs.attention_mask

        wrapped_img_embeds, wrapped_atts_img, _ = self.input_warp(input_ids, attention_mask, input_image=input_image)

        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        # if self.using_prefix_tuning:

        sample_outputs = self.llama_model.generate(
            inputs_embeds=wrapped_img_embeds,
            attention_mask=wrapped_atts_img,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=True,
            force_generation=force_generation
        )

        return sample_outputs

    def prepare_inputs_for_generation(self, input_ids, inputs_embeds=None, force_generation=None, *args, **kwargs):
        model_kwargs = self.base_model_prepare_inputs_for_generation(input_ids, inputs_embeds=inputs_embeds, *args, **kwargs)
        bs = inputs_embeds.shape[0]
        device = inputs_embeds.device
        if 'input_ids' in model_kwargs:
            new_token_ids = model_kwargs['input_ids'][:, -1:]
            if new_token_ids == self.output_img_id:
                #Generated the image token, force add all the image tokens
                current_position_ids = model_kwargs['position_ids'][0, -1] #TODO: Only support batch size 1
                all_img_tokens = torch.tensor(self.llama_tokenizer.convert_tokens_to_ids(ALL_IMG_TOKENS)).unsqueeze(0).to(device)
                all_img_tokens_mask = torch.ones_like(all_img_tokens)[:, :IMG_TOKEN_NUM-1].to(device)
                all_img_position_ids = torch.arange(current_position_ids, current_position_ids + IMG_TOKEN_NUM).unsqueeze(0).to(device)
                
                model_kwargs['attention_mask'] = torch.cat([model_kwargs['attention_mask'], all_img_tokens_mask], dim=1)
                model_kwargs['position_ids'] = all_img_position_ids
                inputs_embeds = self.get_input_embeddings(all_img_tokens)
                model_kwargs['input_ids'] = None
                model_kwargs['inputs_embeds'] = inputs_embeds
                # input_ids = torch.cat([input_ids[:, :-1], all_img_tokens], dim=1)

        if self.using_prefix_tuning:
            if isinstance(self.peft_config, PromptLearningConfig):
                if self.peft_config.peft_type == PeftType.PREFIX_TUNING and model_kwargs["past_key_values"] is None:
                    prefix_attention_mask = torch.ones(bs, self.peft_config.num_virtual_tokens).to(device)
                    model_kwargs["attention_mask"] = torch.cat(
                        (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
                    )

                if model_kwargs["past_key_values"] is None and self.peft_config.peft_type == PeftType.PREFIX_TUNING:
                    past_key_values = self.get_prompt(batch_size=bs, device=device)

                    if self.base_model_torch_dtype is not None:
                        # handle the case for Bloom where it outputs tuple of tuples
                        if isinstance(past_key_values[0], tuple):
                            past_key_values = tuple(
                                tuple(
                                    past_key_value.to(self.base_model_torch_dtype)
                                    for past_key_value in past_key_value_tuple
                                )
                                for past_key_value_tuple in past_key_values
                            )
                        else:
                            past_key_values = tuple(
                                past_key_value.to(self.base_model_torch_dtype) for past_key_value in past_key_values
                            )

                    model_kwargs["past_key_values"] = past_key_values
                else:
                    if model_kwargs["past_key_values"] is None:
                        inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                        prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                        prompts = prompts.to(inputs_embeds.dtype)
                        model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
                        model_kwargs["input_ids"] = None

        return model_kwargs
    
    def get_input_embeddings(self, input_ids):
        if self.using_lora:
            embed_tokens = self.llama_model.base_model.model.model.embed_tokens
        else:
            embed_tokens = self.llama_model.model.embed_tokens
        inputs_embeds = embed_tokens(input_ids)

        return inputs_embeds
    
    def reset_embeddings(self):
        with torch.no_grad():
            if self.using_lora:
                for n, p in self.llama_model.named_parameters():
                    if p.grad is None:
                        continue
                    if "lm_head" in n:
                        p.grad = p.grad*self.output_embed_grad_mask
                    elif "embed_tokens" in n:
                        p.grad = p.grad*self.input_embed_grad_mask
            else:
                self.llama_model.get_input_embeddings().weight.grad = self.llama_model.get_input_embeddings().weight.grad*self.input_embed_grad_mask
                self.llama_model.get_output_embeddings().weight.grad = self.llama_model.get_output_embeddings().weight.grad*self.output_embed_grad_mask
