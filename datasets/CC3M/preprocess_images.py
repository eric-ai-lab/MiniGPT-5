from typing import Any
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from lightning.pytorch import LightningModule, Trainer, seed_everything
from diffusers import AutoencoderKL
from pathlib import Path
from joblib import Parallel, delayed
from constants import *


class CC3MProcessDataset(Dataset):
    def __init__(self, data_path: str, output_vis_processor=None):
        self.output_vis_processor = output_vis_processor
        list_data_table = pd.read_csv(data_path, delimiter='\t')
        self.all_images = list_data_table['image_path'].tolist()

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, i):
        image_path = self.all_images[i]
        path = Path(image_path)
        image_path = Path(CC3M_FOLDER).joinpath(path)
        image_path = str(image_path)
        image = Image.open(image_path).convert("RGB")
        image = self.expand2square(image, (255, 255, 255))

        output_image = self.output_vis_processor(image)

        return {"output_processed_image": output_image, 'image_path': image_path}
    
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

def save_features(output_image_moment, image_path, save_folder):
    image_name = Path(image_path).name
    output_feature_name = image_name.replace('.jpg', '_output.pt')
    output_feature_path = os.path.join(save_folder, output_feature_name)
    torch.save(output_image_moment.detach().cpu(), output_feature_path, pickle_protocol=5)


class Processor(LightningModule):
    def __init__(self, save_folder) -> None:
        super().__init__()
        self.save_folder = save_folder
        self.output_vis_processor = transforms.Compose(
                [
                    transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(512),
                    transforms.ToTensor(),
                ]
            )
        

        sd_model_name = "stabilityai/stable-diffusion-2-1-base"
        self.output_model = AutoencoderKL.from_pretrained(sd_model_name, subfolder="vae").to(PRECISION)
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        output_processed_image = batch["output_processed_image"].to(self.device,PRECISION)
        image_path = batch["image_path"]

        with torch.no_grad():
            h = self.output_model.encoder(output_processed_image)
            output_image_moments = self.output_model.quant_conv(h)
        
        Parallel(n_jobs=25)(delayed(save_features)(output_image_moments[i], image_path[i], self.save_folder) for i in range(len(image_path)))

if __name__ == "__main__":
    seed_everything(42)
    process_file = os.path.join(CC3M_FOLDER, 'cc3m_train.tsv')
    save_folder = os.path.join(CC3M_FOLDER, 'processed_features', 'train')

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    processor = Processor(save_folder)
    processor.eval()
    processor.freeze()

    devices = [0,1,2,3]
    prediction_manager = Trainer(
                        max_epochs=1,
                        accelerator="gpu", devices=devices, 
                        strategy='ddp', 
                        logger = None,
                        precision='bf16-mixed')
    output_vis_processor = processor.output_vis_processor

    dataset = CC3MProcessDataset(process_file, output_vis_processor)
    dataloader = DataLoader(dataset, batch_size=50, num_workers=32, shuffle=False)
    prediction_manager.predict(processor, dataloaders=dataloader, return_predictions=False)