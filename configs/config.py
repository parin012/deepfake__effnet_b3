#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import torch

TRAIN_ROOTS = [
    "/content/DFD_Cropped_Frames/Dataset/Train",
    "/content/data_deepfake/deepfake-and-real-images/Dataset/Train",
]
VAL_ROOTS = [
    "/content/DFD_Cropped_Frames/Dataset/Validation",
    "/content/data_deepfake/deepfake-and-real-images/Dataset/Validation",
]

CROP_CACHE_DIR = "/content/drive/MyDrive/face_cache_224"

OUT_DIR = "/content/drive/MyDrive/experiments/deepfake_b3_aug"

# ---------- 모델 / 학습 하이퍼파라미터 ----------
MODEL_NAME = "tf_efficientnet_b3_ns"
IMG_SIZE = 224
BATCH_SIZE = 160
EPOCHS = 80
WARMUP_EPOCHS = 2
EMA_DECAY = 0.999
GAMMA = 2.0          
PATIENCE = 10

# LR
LR_BACKBONE = 1e-4
LR_HEAD = 3e-4

# 증강 / TTA
USE_TTA_HFLIP = True
LIGHT_AUG_TRAIN = True
SKIP_IF_NO_CACHE = True   # 캐시 없는 이미지는 스킵


SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


INIT_WEIGHTS = None   
INIT_META = None      

