#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from configs.config import CROP_CACHE_DIR


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def set_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gather_image_pairs(roots):
    if not isinstance(roots, (list, tuple)):
        roots = [roots]
    pairs = []
    for base in roots:
        for cls, lbl in [("Real", 0), ("Fake", 1)]:
            d = Path(base) / cls
            if not d.is_dir():
                continue
            for ext in IMG_EXTS:
                for p in d.glob(f"*{ext}"):
                    pairs.append((str(p), lbl))
    return pairs


def cache_path_for(src_path: str) -> str:
    src = Path(src_path)
    cls = src.parent.name  # Real or Fake

    stem = src.stem
    parts = stem.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        vid = "_".join(parts[:-1])
    else:
        vid = src.parent.name

    out_dir = Path(CROP_CACHE_DIR) / cls / vid
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / (stem + ".jpg"))


def macro_f1(probs: np.ndarray, ys: np.ndarray, thr: float) -> float:
    preds = (probs >= thr).astype(int)
    return f1_score(ys, preds, average="macro")


def find_best_threshold(probs: np.ndarray, ys: np.ndarray,
                        low: float = 0.2, high: float = 0.8, steps: int = 25):
    best_t, best_s = 0.5, -1
    for t in np.linspace(low, high, steps):
        s = macro_f1(probs, ys, t)
        if s > best_s:
            best_s, best_t = s, t
    return best_t, best_s



def _atomic_save(obj, path: str):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=d, delete=False) as f:
        tmp = f.name
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_best_model(ema_model, meta: dict, out_dir: str):
    ckpt_best = os.path.join(out_dir, "best_model.pt")
    meta_best = os.path.join(out_dir, "best_meta.json")
    _atomic_save(ema_model.state_dict(), ckpt_best)
    with open(meta_best, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def save_last(model, ema_model, optimizer, epoch: int, step: int, out_dir: str):
    ckpt_last = os.path.join(out_dir, "last_model.pt")
    ckpt_full = os.path.join(out_dir, "last_full.pt")

    _atomic_save(ema_model.state_dict(), ckpt_last)
    pkg = {
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    _atomic_save(pkg, ckpt_full)

