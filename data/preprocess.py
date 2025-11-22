#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

from insightface.app import FaceAnalysis
import torchvision.transforms as T

from configs.config import (
    CROP_CACHE_DIR, IMG_SIZE,
    TRAIN_ROOTS, VAL_ROOTS, LIGHT_AUG_TRAIN
)
from train.utils import gather_image_pairs


DET_SIZES = [(640, 640), (896, 896)]
CONF_TH = 0.40
MARGIN = 1.30
MIN_FACE_FRAC = 0.12   
MIN_BLUR_VAR = 20.0    

app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


def _detect_best_face(img_bgr):
    H, W = img_bgr.shape[:2]
    cx, cy = W / 2, H / 2

    for ds in DET_SIZES:
        app.prepare(ctx_id=0, det_size=ds)
        faces = app.get(img_bgr)
        faces = [f for f in faces if float(getattr(f, "det_score", 0.0)) >= CONF_TH]
        if not faces:
            continue

        best = None
        best_score = -1
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            area = ((x2 - x1) * (y2 - y1)) / (W * H + 1e-6)
            fx, fy = (x1 + x2) / 2, (y1 + y2) / 2
            d_center = np.hypot(fx - cx, fy - cy) / max(W, H)
            score = float(f.det_score) * (0.6 * area + 0.4 * (1.0 / (1.0 + d_center)))
            if score > best_score:
                best_score = score
                best = f

        if best is not None:
            return best

    return None


def crop_face(img_pil):
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    best = _detect_best_face(img_bgr)
    if best is None:
        return None, {"reason": "no_detect"}

    x1, y1, x2, y2 = best.bbox.astype(int)
    face_frac = ((x2 - x1) * (y2 - y1)) / (W * H + 1e-6)
    if face_frac < MIN_FACE_FRAC:
        return None, {"reason": "small_face"}

    # margin 적용
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    w = int((x2 - x1) * MARGIN)
    h = int((y2 - y1) * MARGIN)

    x1 = max(0, cx - w // 2)
    y1 = max(0, cy - h // 2)
    x2 = min(W, cx + w // 2)
    y2 = min(H, cy + h // 2)

    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None, {"reason": "empty_roi"}

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lv = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lv < MIN_BLUR_VAR:
        return None, {"reason": "blur", "lap_var": float(lv)}

    face = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return Image.fromarray(face_rgb), {
        "lap_var": float(lv),
        "det_score": float(best.det_score)
    }


def build_face_cache(roots):
    if not isinstance(roots, (list, tuple)):
        roots = [roots]

    os.makedirs(CROP_CACHE_DIR, exist_ok=True)
    images = gather_image_pairs(roots)
    kept, skipped = 0, 0

    for src_path, lbl in tqdm(images, desc="build_face_cache"):
        src = Path(src_path)
        cls = src.parent.name  

        stem = src.stem
        parts = stem.split("_")
        if len(parts) > 1 and parts[-1].isdigit():
            vid = "_".join(parts[:-1])
        else:
            vid = src.parent.name

        out_dir = Path(CROP_CACHE_DIR) / cls / vid
        out_dir.mkdir(parents=True, exist_ok=True)

        dst_img = out_dir / (stem + ".jpg")
        dst_meta = out_dir / (stem + ".json")

        if dst_img.exists():
            continue

        try:
            img = Image.open(src).convert("RGB")
        except (UnidentifiedImageError, OSError, FileNotFoundError):
            skipped += 1
            continue

        face, meta = crop_face(img)
        if face is None:
            skipped += 1
            meta = meta or {}
            meta.update({"src": str(src)})
            with open(dst_meta, "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            continue

        face.save(dst_img, quality=95)
        meta = meta or {}
        meta.update({"src": str(src)})
        with open(dst_meta, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        kept += 1

    print(f"[face_cache] kept={kept}, skipped={skipped}, dir={CROP_CACHE_DIR}")



import random
import io


class RandomJPEG:
    def __init__(self, qmin=40, qmax=90, p=0.7):
        self.qmin, self.qmax, self.p = qmin, qmax, p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        q = random.randint(self.qmin, self.qmax)
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomGaussianNoise:
    def __init__(self, sigma_min=1.0, sigma_max=6.0, p=0.5):
        self.smin, self.smax, self.p = sigma_min, sigma_max, p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        x = np.asarray(img).astype(np.float32)
        sigma = random.uniform(self.smin, self.smax)
        noise = np.random.normal(0, sigma, x.shape).astype(np.float32)
        y = np.clip(x + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(y)


def get_transforms(is_train: bool, img_size: int = IMG_SIZE):
    if is_train and LIGHT_AUG_TRAIN:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
            RandomJPEG(40, 90, p=0.7),
            RandomGaussianNoise(1.0, 6.0, p=0.5),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.25),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.1, 0.1, 0.1, 0.03),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])


if __name__ == "__main__":
    all_roots = TRAIN_ROOTS + VAL_ROOTS
    build_face_cache(all_roots)

