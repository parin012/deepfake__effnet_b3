#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import argparse

from PIL import Image
import torch
import torchvision.transforms as T

from configs.config import OUT_DIR, DEVICE, IMG_SIZE
from models.efficientnet_b3 import build_model


def load_model_and_meta():
    ckpt_path = os.path.join(OUT_DIR, "best_model.pt")
    meta_path = os.path.join(OUT_DIR, "best_meta.json")

    if not os.path.exists(ckpt_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("best_model.pt / best_meta.json not found in OUT_DIR")

    meta = json.load(open(meta_path))
    best_thr = float(meta.get("best_thr", 0.5))
    mean = meta.get("mean", [0.485, 0.456, 0.406])
    std = meta.get("std", [0.229, 0.224, 0.225])

    model = build_model()
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd)
    model.to(DEVICE).eval()

    tfm = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    return model, tfm, best_thr


def predict_image(path: str):
    model, tfm, thr = load_model_and_meta()
    img = Image.open(path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logit = model(x)
        prob_fake = torch.sigmoid(logit).item()

    pred = int(prob_fake >= thr)
    cls_name = "Fake" if pred == 1 else "Real"
    return prob_fake, cls_name, thr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="path to input image")
    args = parser.parse_args()

    prob_fake, cls_name, thr = predict_image(args.image)
    print(f"Image: {args.image}")
    print(f"Prob(Fake) = {prob_fake:.4f} (thr={thr:.3f}) → Pred = {cls_name}")


if __name__ == "__main__":
    main()

