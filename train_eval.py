import os
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field

import torch
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
import transformers
from torch.utils.data import DataLoader
from typing import Optional, Dict
from dataclasses import dataclass, field
from dataloader import SupervisedDataset, DataCollator
from lightning.pytorch.loggers import WandbLogger

from lightning.pytorch import seed_everything
from torchvision import transforms
from constants import *

from lightning.pytorch.callbacks import BasePredictionWriter
from model import MiniGPT5_Model, MiniGPT5_InputProcessor
from metric import *

class PredWriter(BasePredictionWriter):
    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: Any,  # complex variables is ok
        batch_indices: list[list[list[int]]],
    ) -> None:
        output_folder = pl_module.output_folder
        torch.save(predictions, os.path.join(output_folder, f"predictions-{trainer.local_rank}.pt"))
        print(f'rank {trainer.local_rank} predictions saved')

def default_gpus():
    return [0,1,2,3]

@dataclass
class ModelArguments:
    model_type: Optional[str] = field(default="multimodal_encoder")
    snr_loss: Optional[bool] = field(default=True)
    model_save_name: Optional[str] = field(default="model_{epoch}-{step}")
    stage1_weight: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    val_data_path: str = field(default=None, metadata={"help": "Path to the validation data."})
    test_data_path: str = field(default=None, metadata={"help": "Path to the test data."})

@dataclass
class TrainingArguments:
    cache_dir: Optional[str] = field(default=None)
    output_dir: str = field(default=WEIGHTFOLDER)
    num_train_epochs:int = field(default=2)
    per_device_train_batch_size:int = field(default=2)
    per_device_eval_batch_size:int = field(default=2)
    real_batch_size:int = field(default=48)
    save_total_limit:int = field(default=1)
    learning_rate:float = field(default=2e-5)
    warmup_ratio:float = field(default=0.03)
    warmup_steps:int = field(default=1000)
    adam_epsilon:float = field(default=1e-8)

    num_workers:int = field(default=16)

    gpus: List[int] = field(default_factory=default_gpus)
    resume: Optional[str] = field(default=None)
    is_training: Optional[bool] = field(default=False)
    test_weight: Optional[str] = field(default=None)
        
def make_supervised_data_module(data_args, training_args, data_collator, input_processor=None, output_vis_processor=None) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    train_dataset = SupervisedDataset(data_path=data_args.train_data_path, input_processor=input_processor, output_vis_processor=output_vis_processor)
    eval_dataset = SupervisedDataset(data_path=data_args.val_data_path, input_processor=input_processor, output_vis_processor=output_vis_processor)

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=training_args.per_device_train_batch_size,
                                    num_workers=training_args.num_workers,
                                  collate_fn=data_collator,
                                  prefetch_factor=4,
                                  pin_memory=True)
    val_dataloader = DataLoader(eval_dataset, 
                                batch_size=training_args.per_device_eval_batch_size,
                                num_workers=training_args.num_workers,
                                collate_fn=data_collator,
                                prefetch_factor=4,
                                pin_memory=True)
    return train_dataloader, val_dataloader


def make_eval_data_module(data_args, training_args, data_collator, input_processor=None, output_vis_processor=None) -> Dict:
    eval_dataset = SupervisedDataset(data_path=data_args.test_data_path, input_processor=input_processor, output_vis_processor=output_vis_processor, test=True)
    val_dataloader = DataLoader(eval_dataset, batch_size=1, collate_fn=data_collator)
    return val_dataloader

if __name__ == "__main__":
    seed_everything(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if isinstance(training_args.gpus, str):
        training_args.gpus = [int(x) for x in training_args.gpus.split(',')]
    if isinstance(data_args.train_data_path, str):
        data_args.train_data_path = os.path.join(DATAFOLDER, data_args.train_data_path)
    if isinstance(data_args.val_data_path, str):
        data_args.val_data_path = os.path.join(DATAFOLDER, data_args.val_data_path)
    if isinstance(data_args.test_data_path, str):
        data_args.test_data_path = os.path.join(DATAFOLDER, data_args.test_data_path)

    batch_size = training_args.real_batch_size
    devices = training_args.gpus
    num_devices = len(devices)
    gradient_accumulation_steps = max(1,batch_size // (training_args.per_device_train_batch_size*num_devices))

    if IS_STAGE2:
        stage1_weight = model_args.stage1_weight
        assert stage1_weight is not None, "stage2 weight needs stage1 weight, but stage1 weight path is None"
        stage1_weight = os.path.join(WEIGHTFOLDER, stage1_weight)
        model = MiniGPT5_Model.load_from_checkpoint(stage1_weight, strict=False, map_location="cpu", encoder_model_config=model_args, **vars(training_args))
    else:
        model = MiniGPT5_Model(encoder_model_config=model_args, **vars(training_args))
    
    tokenizer = model.tokenizer
    sd_tokenizer = model.sd_tokenizer

    if training_args.is_training:

        output_vis_processor = transforms.Compose(
                [
                    transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(512),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
        input_vis_processor = transforms.Compose(
                [
                    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ]
            )
        input_processor = MiniGPT5_InputProcessor(tokenizer=tokenizer, image_processor=input_vis_processor)

        data_collator = DataCollator(tokenizer=tokenizer, sd_tokenizer=sd_tokenizer)

        train_dataloader, val_dataloader = make_supervised_data_module(data_args, training_args, data_collator, input_processor, output_vis_processor)
        
        checkpoint_callback = ModelCheckpoint(
                dirpath=training_args.output_dir,
                filename=model_args.model_save_name,
                monitor="val_loss",
                save_top_k=1,
                # save_last=True,
            )
        
        strategy = 'ddp'
        if "CC3M" in DATAFOLDER:
            val_check_interval = 0.25
        else:
            val_check_interval = 0.5
            strategy = 'ddp_find_unused_parameters_true'

        wandb_logger = WandbLogger(save_dir=training_args.output_dir, project="MiniGPT5_Model", offline=False, name=model_args.model_save_name)
        trainer = Trainer(default_root_dir=training_args.output_dir, max_epochs=training_args.num_train_epochs, 
                        accumulate_grad_batches=gradient_accumulation_steps,
                        accelerator="gpu", devices=devices, 
                        strategy=strategy,
                        logger = wandb_logger, 
                        precision='bf16-mixed',
                        val_check_interval=val_check_interval,
                        callbacks=[checkpoint_callback])
        resume = training_args.resume
        trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=resume)
    else:
        # model.image_pipeline.enable_xformers_memory_efficient_attention()
        output_vis_processor = transforms.Compose(
                [
                    transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(512),
                    transforms.ToTensor(),
                ]
            )
        input_vis_processor = transforms.Compose(
                [
                    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ]
            )

        input_processor = MiniGPT5_InputProcessor(tokenizer=tokenizer, image_processor=input_vis_processor)

        data_collator = DataCollator(tokenizer=tokenizer, sd_tokenizer=sd_tokenizer)

        pred_writer = PredWriter(write_interval="epoch")

        wandb_logger = WandbLogger(save_dir=training_args.output_dir, project="MiniGPT5_Model", offline=True, name=model_args.model_save_name)
        trainer = Trainer(default_root_dir=training_args.output_dir, max_epochs=training_args.num_train_epochs, 
                        accelerator="gpu", devices=devices, 
                        strategy='ddp', 
                        logger = wandb_logger, 
                        precision='bf16-mixed',
                        callbacks=[pred_writer])
        
        val_dataloader = make_eval_data_module(data_args, training_args, data_collator, input_processor, output_vis_processor)
        assert training_args.test_weight is not None, "test weight path is None"
        ckpt_path = os.path.join(WEIGHTFOLDER, training_args.test_weight)
        model.output_folder = os.path.join(OUTPUT_FOLDER, training_args.test_weight.split(".")[0]+OUTPUT_SUFFIX)
        if not os.path.exists(model.output_folder):
            os.makedirs(model.output_folder)
        trainer.predict(model, val_dataloader, return_predictions=False, ckpt_path=ckpt_path)
