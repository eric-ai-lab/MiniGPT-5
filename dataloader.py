from torch.utils.data import Dataset
from typing import Optional, Dict, Sequence
import transformers
from tqdm import tqdm
from dataclasses import dataclass
import pandas as pd
import re
import os
import numpy as np
from PIL import Image
import random
from pathlib import Path
import json
import copy
from constants import *

class CC3MDataset(Dataset):
    def __init__(self, data_path: str, input_processor=None, output_vis_processor=None, test=False):
        self.test = test
        self.input_processor = input_processor
        self.output_vis_processor = output_vis_processor
        self.output_img_id = input_processor.tokenizer.convert_tokens_to_ids(ALL_IMG_TOKENS[0])
        self.load_preprocessed_image_features = not test
        saved_data_path = data_path.replace('.tsv', '_8-token_stage1.pkl')
        if os.path.exists(saved_data_path):
            print("Loading saved data...")
            self.recover_data(saved_data_path)
            print("Loaded saved data for CC3M")
        else:
            list_data_table = pd.read_csv(data_path, delimiter='\t')
            self.sources, self.targets, self.input_image_path, self.output_image_path = [], [], [], []
            self.caption, self.task_names = [], []

            system_prompt="You will be able to generate image according to command."
            generation_prompts = [
                "generate image with caption:",
                "can you give me the image with caption:",
                "help me to generate this image:",
                "generate image with according to caption:",
                "according to caption, generate image:",
                "an image with caption:",
                "can you visualize this caption:",
            ]

            for i in tqdm(range(len(list_data_table))):
                data = list_data_table.iloc[i]
                step_image = data['image_path']
                step_caption = data['caption']
                path = Path(step_image)
                step_image = Path(DATAFOLDER).joinpath(path)
                step_image = str(step_image)
                step_caption = self.pre_caption(step_caption)

                caption_source = f"{step_caption}"
                caption_target = f'{ALL_IMG_TOKENS_STR} ###'
                self.sources.append(caption_source)
                self.targets.append(caption_target)
                self.caption.append(step_caption)
                self.task_names.append(f'cc3m_{i}')
                self.input_image_path.append([None])
                self.output_image_path.append(step_image)
                
                if i%100 == 0 and not test:
                    caption_source = f"###Human: {random.choice(generation_prompts)} {step_caption} ###Assistant:"
                    caption_source = system_prompt + caption_source
                    caption_target = f'{ALL_IMG_TOKENS_STR} ###'
                    self.sources.append(caption_source)
                    self.targets.append(caption_target)
                    self.caption.append(step_caption)
                    self.task_names.append(f'cc3m_{i}_instruction')
                    self.input_image_path.append([None])
                    self.output_image_path.append(step_image)
            self.valid_idx = list(range(len(self.sources)))
            print("Saving data...")
            self.save_process_data(saved_data_path)
            print("Saved data for cc3m!")
        if test:
            self.targets = self.caption

    def recover_data(self, saved_file):
        all_data = torch.load(saved_file)
        self.sources = all_data['sources']
        self.targets = all_data['targets']
        self.input_image_path = all_data['input_image_path']
        self.output_image_path = all_data['output_image_path']
        self.caption = all_data['caption']
        self.task_names = all_data['task_names']
        del all_data
        if self.test:
            self.valid_idx = []
            for i in range(len(self.targets)):
                if self.output_image_path[i] is not None:
                    self.valid_idx.append(i)
            

    def save_process_data(self, saved_file):
        all_data = {'sources': self.sources,
                    'targets': self.targets,
                    'input_image_path': self.input_image_path,
                    'output_image_path': self.output_image_path,
                    'caption': self.caption,
                    'task_names': self.task_names,
                    }
        torch.save(all_data, saved_file)
    
    def __len__(self):
        if self.test:
            return len(self.valid_idx)
        return len(self.sources)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.test:
            i = self.valid_idx[i]
        input_image_path = self.input_image_path[i]
        output_image_path = self.output_image_path[i]
        input_text = self.sources[i]
        output_text = self.targets[i]
        if self.load_preprocessed_image_features and PREPROCESS_FEATURE_FOLDER is not None and os.path.isdir(PREPROCESS_FEATURE_FOLDER):
            if output_image_path is not None:
                output_feature_name = Path(output_image_path).name
                output_feature_name = output_feature_name.replace('.jpg', '_output.pt')
                if 'val' in output_image_path:
                    output_feature_path = Path(PREPROCESS_FEATURE_FOLDER).joinpath('val', output_feature_name)
                elif 'train' in output_image_path:
                    output_feature_path = Path(PREPROCESS_FEATURE_FOLDER).joinpath('train', output_feature_name)
                output_image_feature = torch.load(output_feature_path).unsqueeze(0)
            else:
                output_image_path = 'none'
                output_image_feature = torch.zeros((1, 8, 64, 64))
            
            input_images_feature = []
            for in_img_path in input_image_path:
                if in_img_path is not None:
                    input_feature_name = Path(in_img_path).name
                    input_feature_name = input_feature_name.replace('.jpg', '_input.pt')
                    if 'val' in in_img_path:
                        input_feature_path = Path(PREPROCESS_FEATURE_FOLDER).joinpath('val', input_feature_name)
                    elif 'train' in in_img_path:
                        input_feature_path = Path(PREPROCESS_FEATURE_FOLDER).joinpath('train', input_feature_name)
                    input_image_feature = torch.load(input_feature_path).unsqueeze(0)
                else:
                    input_image_feature = torch.zeros((1, 32, 4096))
                input_images_feature.append(input_image_feature)
            input_images_feature = torch.cat(input_images_feature, dim=0)
            input_dict = self.input_processor(text = input_text, add_special_tokens=False)
            input_dict['input_images_feature'] = input_images_feature
            input_dict['output_image_feature'] = output_image_feature
        else:
            input_images = []
            for in_img_path in input_image_path:
                if in_img_path is not None:
                    input_image = Image.open(in_img_path).convert("RGB")
                else:
                    input_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
                input_images.append(input_image)
            input_dict = self.input_processor(text = input_text, images = input_images, add_special_tokens=False)
            input_dict['original_images'] = input_images
            
            if output_image_path is not None:
                output_image = Image.open(output_image_path).convert("RGB")
                output_image = self.expand2square(output_image, (255, 255, 255))
                output_image = self.output_vis_processor(output_image)
                output_image = output_image.unsqueeze(0)
            else:
                output_image_path = 'none'
                output_image = torch.zeros((1, 3, 512, 512))
            input_dict["output_image"] = output_image

        input_dict["caption"] = self.caption[i]
        input_dict["task_name"] = self.task_names[i]
        target_ids = self.input_processor(text = output_text, add_special_tokens=False)['input_ids']
        label = torch.ones_like(input_dict["input_ids"])*-100
        label = torch.cat((label, target_ids), dim=1)
        index = torch.nonzero(label == self.output_img_id)
        if len(index):
            index = index[0,1]
            label[:, index+1:index+IMG_TOKEN_NUM-1] = -100
        input_dict["labels"] = label
        input_dict["input_ids"] = torch.cat((input_dict["input_ids"], target_ids), dim=1)
        input_dict["attention_mask"] = torch.cat((input_dict["attention_mask"], torch.ones_like(target_ids)), dim=1)
        input_dict["source"] = input_text
        input_dict["target"] = output_text

        return input_dict

    def pre_caption(self, caption):
        
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        # max_words = 100
        # caption_words = caption.split(" ")
        # if len(caption_words) > max_words:
        #     caption = " ".join(caption_words[: max_words])

        return caption
    
    @staticmethod
    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

class VISTDataset(CC3MDataset):
    def __init__(self, data_path: str, input_processor=None, output_vis_processor=None, test=False):
        self.test = test
        self.input_processor = input_processor
        self.output_vis_processor = output_vis_processor
        self.output_img_id = input_processor.tokenizer.convert_tokens_to_ids(ALL_IMG_TOKENS[0])
        eos_token = input_processor.tokenizer.eos_token
        self.load_preprocessed_image_features = False

        self.sources, self.targets, self.input_image_path, self.output_image_path = [], [], [], []
        self.caption, self.task_names = [], []
        all_tasks = json.load(open(data_path, 'r'))
        image_id_mapping = json.load(open(data_path.replace('cleaned', 'image_mapping'), 'r'))
        system_prompt1="Give the following images in <Img>ImageContent</Img> format. "\
           "You will be able to see the images once I provide it to you. Please understanding images and generate story."
        human_prompts1 = [
            "###Human:{prompt} Generate an image with the scene description: {step_text} ###Assistant:",
            "###Human:{prompt} the scene description: {step_text} ###Assistant:",
        ]
        human_prompts2 = [
            "###Human:{prompt} Tell me the next scene with image. ###Assistant:",
            "###Human:{prompt} Generate the next scene with image. ###Assistant:",
            "###Human:{prompt} What should happen then? ###Assistant:"
        ]
        human_prompts3 = [
            "###Human:{prompt} Tell me the next scene description by this image: <Img><ImageHere></Img> ###Assistant:",
            "###Human:{prompt} What happen in the next scene image: <Img><ImageHere></Img> ###Assistant:"
        ]
        for task_name, task in tqdm(all_tasks.items()):
            task_prompts = []
            task_input_image_path = []
            for step in task:
                step_text = step['caption']
                step_image = step['image_id']
                sequence_index = step['sequence_index']
                step_image = os.path.join(DATAFOLDER, image_id_mapping[step_image])

                prompt = "<Img><ImageHere></Img>".join(task_prompts)
                if len(task_prompts):
                    prompt = f"{prompt}<Img><ImageHere></Img>\n"

                step_input_image = copy.deepcopy(task_input_image_path)
                step_name = f"{task_name}_{sequence_index}"
                #image generation
                step_source = random.choice(human_prompts1).format(prompt=prompt, step_text=step_text)
                step_source = system_prompt1 + step_source
                step_target = f"{ALL_IMG_TOKENS_STR} ###"
                self.sources.append(step_source)
                self.caption.append(None)
                self.targets.append(step_target)
                self.task_names.append(step_name+"-gen")
                if len(step_input_image):
                    self.input_image_path.append(step_input_image)
                else:
                    self.input_image_path.append([None])
                self.output_image_path.append(step_image)

                if len(task_prompts) and not test:
                    #image and text generation
                    step_source = random.choice(human_prompts2).format(prompt=prompt)
                    step_source = system_prompt1 + step_source
                    step_target = f"{step_text} {ALL_IMG_TOKENS_STR} ###"
                    self.sources.append(step_source)
                    self.targets.append(step_target)
                    self.caption.append(None)
                    self.task_names.append(step_name+"-multimodal")
                    self.input_image_path.append(step_input_image)
                    self.output_image_path.append(step_image)
                    
                    #image understanding
                    step_source = random.choice(human_prompts3).format(prompt=prompt)
                    step_source = system_prompt1 + step_source
                    step_target = f"{step_text} ###"
                    self.sources.append(step_source)
                    self.targets.append(step_target)
                    self.caption.append(None)
                    self.task_names.append(step_name+"-understanding")
                    self.input_image_path.append(step_input_image+[step_image])
                    self.output_image_path.append(None)

                task_prompts.append(step_text)
                task_input_image_path.append(step_image)

        self.valid_idx = list(range(len(self.sources)))
        print('Load data done!')

class MMDialogDataset(CC3MDataset):
    def __init__(self, data_path: str, input_processor=None, output_vis_processor=None, test=False):
        self.test = test
        self.input_processor = input_processor
        self.output_vis_processor = output_vis_processor
        self.output_img_id = input_processor.tokenizer.convert_tokens_to_ids(ALL_IMG_TOKENS[0])
        eos_token = input_processor.tokenizer.eos_token
        self.load_preprocessed_image_features = False
        
        system_prompt="Give the following images in <Img>ImageContent</Img> format. "\
           "You will be able to see the images once I provide it to you. Please generate conversation with appropriate image."
        
        self.sources, self.targets, self.input_image_path, self.output_image_path = [], [], [], []
        self.caption, self.task_names = [], []
        data_folder = os.path.dirname(data_path)
        with open(data_path, 'r') as f:
            all_data = f.readlines()
        for data in tqdm(all_data): 
            data = json.loads(data)
            data_num = data['conversation_id']
            conversation = data['conversation']
            if len(conversation)<2:
                continue
            history_prompt = system_prompt
            history_images = []
            for i, conv in enumerate(conversation):
                turn = conv['turn']
                turn_text = turn[0]['__TEXT__']
                if len(turn)==1:
                    turn_image_path = None
                else:
                    turn_image_path = os.path.join(data_folder, f"{turn[1]['__MEDIA__']}.jpg")
                    if not os.path.exists(turn_image_path):
                        # print(f'Cannot Find: {turn_image_path}')
                        break
                if len(turn_text.split(' '))>20 and not test:
                    break
                if i%2==0:
                    source = history_prompt + "###Human:"
                else:
                    source = history_prompt + "###Assistant:"

                if i>0:
                    self.sources.append(source)
                    if turn_image_path is not None:
                        target = f"{turn_text} {ALL_IMG_TOKENS_STR} ###"
                    else:
                        target = f"{turn_text} ###"
                    self.targets.append(target)
                    self.caption.append(None)
                    self.task_names.append(f'mmdialog{data_num}_{i}')
                    if len(history_images):
                        self.input_image_path.append(copy.deepcopy(history_images))
                    else:
                        self.input_image_path.append([None])
                    self.output_image_path.append(turn_image_path)

                if turn_image_path is not None:
                    history_prompt = source + f" {turn_text} <Img><ImageHere></Img>\n"
                    history_images.append(turn_image_path)
                else:
                    history_prompt = source + f" {turn_text}\n"
                
                if (i%2==1 and i>2):
                    pattern = "###Human:(.*?)###Human:"
                    match = re.search(pattern, history_prompt, re.DOTALL)
                    match_text = match.group(0)
                    history_prompt = history_prompt.replace(match_text, "###Human:")
                    pattern2 = '<Img><ImageHere></Img>'
                    match_num = len(re.findall(pattern2, match_text))
                    if match_num>0:
                        history_images = history_images[match_num:]

        self.valid_idx = list(range(len(self.sources)))
        print('Load data done with {} samples!'.format(len(self.sources)))

    def __getitem__(self, i):
        for _ in range(10):
            try:
                item = super().__getitem__(i)
                break
            except:
                i = random.choice(self.valid_idx)
        return item
    
@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    sd_tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        key_list = instances[0].keys()
        output_dict = {}
        for key in key_list:
            # Need to remove the batch dimension
            if key in ['input_ids', 'attention_mask', 'labels']:
                output_value = [instance[key][0] for instance in instances]
            else:
                output_value = [instance[key] for instance in instances]
            if key == "input_ids":
                output_value = torch.nn.utils.rnn.pad_sequence(output_value, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            elif key == "labels":
                output_value = torch.nn.utils.rnn.pad_sequence(output_value, batch_first=True, padding_value=-100)
            elif key == "attention_mask":
                output_value = torch.nn.utils.rnn.pad_sequence(output_value, batch_first=True, padding_value=0)
            elif key == 'input_images':
                output_value = [v.to(PRECISION) for v in output_value]
            elif key == 'output_image':
                output_value = torch.concat(output_value).to(PRECISION)
            elif key == 'output_image_feature':
                output_value = torch.concat(output_value)
            output_dict[key] = output_value
        return output_dict

if 'CC3M' in DATAFOLDER:
    SupervisedDataset = CC3MDataset
elif 'MMDialog' in DATAFOLDER:
    SupervisedDataset = MMDialogDataset
else:
    SupervisedDataset = VISTDataset
