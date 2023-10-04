import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from torchvision import transforms
from model import MiniGPT5_Model
from train_eval import ModelArguments, DataArguments, TrainingArguments
from PIL import Image
import transformers
import torch
import matplotlib.pyplot as plt
import textwrap
from lightning.pytorch import seed_everything

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
    
if __name__ == "__main__":
    seed_everything(42)
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if isinstance(training_args.gpus, str):
        training_args.gpus = [int(x) for x in training_args.gpus.split(',')]

    stage1_ckpt = model_args.stage1_weight
    stage2_ckpt = training_args.test_weight

    minigpt5 = MiniGPT5_Model.load_from_checkpoint(stage1_ckpt, strict=False, map_location="cpu", encoder_model_config=model_args, **vars(training_args))
    finetuned_state_dict = torch.load(stage2_ckpt, map_location="cpu")['state_dict']
    minigpt5.load_state_dict(finetuned_state_dict, strict=False)
    minigpt5.to("cuda:0")
    minigpt5.eval()
    input_images = None
    input_image_path = [os.path.join(current_dir, '000000005.jpg'), os.path.join(current_dir, '000000007.jpg')]

    if input_image_path:
        input_vis_processor = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )
        input_images = []
        for img_path in input_image_path:
            input_image = Image.open(img_path).convert("RGB")
            input_image = expand2square(input_image, (255, 255, 255))
            input_image = input_vis_processor(input_image)
            input_image = input_image.unsqueeze(0).to("cuda:0")
            input_images.append(input_image)
        input_images = torch.cat(input_images, dim=0)

    system_prompt="Give the following images in <Img>ImageContent</Img> format. "\
           "You will be able to see the images once I provide it to you. Please understanding images and generate story."
    utterance = "my sister arrived early to help me with the family bar bq.<Img><ImageHere></Img>every one else arrived soon after.<Img><ImageHere>\n"
    utterance = system_prompt + f"###Human:{utterance} Tell me the next scene with image. ###Assistant:"

    for i in range(5):
        with torch.inference_mode():
            with torch.autocast("cuda"):
                text_out, image_out = minigpt5.generate(utterance, input_images)
        fig, ax = plt.subplots()
        ax.imshow(image_out)
        generated_text = text_out.replace("###", "").replace("[IMG0]", "")
        wrapped_generated_text = textwrap.fill(generated_text, 50)
        ax.set_title(wrapped_generated_text, pad=20)
        ax.axis('off')
        plt.savefig(os.path.join(current_dir, f'test_{i}.png'), bbox_inches='tight')
        plt.close(fig)