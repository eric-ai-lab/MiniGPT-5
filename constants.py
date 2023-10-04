import torch
import os

CC3M_FOLDER = os.getenv("CC3M_FOLDER", "datasets/CC3M")
DATAFOLDER = os.getenv("DATAFOLDER", "datasets/CC3M")
PREPROCESS_FEATURE_FOLDER = os.getenv("PREPROCESS_FEATURE_FOLDER", None)

WEIGHTFOLDER = os.getenv("WEIGHTFOLDER", "weights")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "outputs")
OUTPUT_SUFFIX = os.getenv("OUTPUT_SUFFIX", "")

IS_STAGE2 = os.getenv("IS_STAGE2", False)
IS_STAGE2 = True if IS_STAGE2 in ["True", "true"] else False

IMG_TOKEN_NUM = 8
ALL_IMG_TOKENS = [f"[IMG{i}]" for i in range(IMG_TOKEN_NUM)]
ALL_IMG_TOKENS_STR = "".join(ALL_IMG_TOKENS)

USE_PREFIX_TUNING = False
USE_LORA = False
USE_CFG = True

if IS_STAGE2:
    USE_LORA = True

PRECISION = torch.bfloat16
TRAINABLE_PRECISION = torch.float32