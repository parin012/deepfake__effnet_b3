#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import amp

from configs.config import (
    OUT_DIR, DEVICE, EPOCHS, WARMUP_EPOCHS, PATIENCE,
    USE_TTA_HFLIP, INIT_WEIGHTS, INIT_META,
    LR_BACKBONE, LR_HEAD, SEED, IMG_SIZE
)
from data.dataset import create_dataloaders
from models.efficientnet_b3 import build_model, FocalLoss, ModelEMA
from train.utils import (
    set_seed, find_best_threshold,
    save_best_model, save_last
)


def tta_logits(model, x):
    logits = model(x)
    if USE_TTA_HFLIP:
        logits = 0.5 * (logits + model(torch.flip(x, dims=[3])))
    return logits


def load_init_weights(model):
    if INIT_WEIGHTS and os.path.exists(INIT_WEIGHTS):
        print(f"[init] loading weights from {INIT_WEIGHTS}")
        sd = torch.load(INIT_WEIGHTS, map_location="cpu")
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[init] missing keys({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"[init] unexpected keys({len(unexpected)}): {unexpected[:5]}")


def build_optimizer_and_scheduler(model):
    head = model.get_classifier() if hasattr(model, "get_classifier") else list(model.children())[-1]
    head_params = list(head.parameters()) if hasattr(head, "parameters") else []
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]

    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params, "lr": LR_HEAD},
        ],
        weight_decay=1e-4,
    )

    MIN_LR_FACTOR = 0.05

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / float(max(1, WARMUP_EPOCHS))
        progress = (epoch - WARMUP_EPOCHS) / float(max(1, EPOCHS - WARMUP_EPOCHS))
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return MIN_LR_FACTOR + (1 - MIN_LR_FACTOR) * cosine

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    return opt, sch


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)

    # ---------- Data ----------
    train_loader, val_loader, train_ds, val_ds = create_dataloaders()
    n_real = sum(lbl == 0 for _, lbl in train_ds.samples)
    n_fake = sum(lbl == 1 for _, lbl in train_ds.samples)
    pos_ratio = n_fake / max(1, n_real + n_fake)
    alpha = 1 - pos_ratio
    print(f"[DATA] Train: Real={n_real}, Fake={n_fake}, pos_ratio={pos_ratio:.3f}, alpha={alpha:.3f}")

    # ---------- Model / Loss / EMA ----------
    model = build_model().to(DEVICE).to(memory_format=torch.channels_last)
    load_init_weights(model)

    ema = ModelEMA(model)
    crit = FocalLoss(alpha=alpha)
    crit_val = nn.BCEWithLogitsLoss()

    opt, sch = build_optimizer_and_scheduler(model)

    amp_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    scaler = amp.GradScaler(enabled=(DEVICE == "cuda" and amp_dtype == torch.float16))

    # ---------- Metrics logging ----------
    metrics_path = os.path.join(OUT_DIR, "metrics.csv")
    if os.path.exists(metrics_path):
        os.remove(metrics_path)
    pd.DataFrame(
        columns=["epoch", "train_loss", "val_loss", "val_macroF1", "thr", "lr"]
    ).to_csv(metrics_path, index=False)

    best_f1, best_thr, best_ep = -1.0, 0.5, -1
    if INIT_META and os.path.exists(INIT_META):
        try:
            meta0 = json.load(open(INIT_META))
            best_f1 = float(meta0.get("macro_f1", best_f1))
            best_thr = float(meta0.get("best_thr", best_thr))
        except Exception:
            pass

    epochs_no_improve = 0
    global_step = 0

    # ---------- Train loop ----------
    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_loss, n_tr = 0.0, 0
        pbar = tqdm(train_loader, desc=f"train[{ep}/{EPOCHS}]")

        for x, y in pbar:
            x = x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", dtype=amp_dtype, enabled=(DEVICE == "cuda")):
                logits = model(x)
                loss = crit(logits, y)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            ema.update(model)

            bs = x.size(0)
            tr_loss += loss.item() * bs
            n_tr += bs
            global_step += 1

            pbar.set_postfix(loss=float(loss.item()))

        tr_loss /= max(1, n_tr)

        # ---------- Validation (EMA + TTA) ----------
        model.eval()
        ema_model = ema.ema

        val_loss, n_val = 0.0, 0
        probs, ys = [], []

        with torch.inference_mode():
            for x, y in val_loader:
                x = x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
                y = y.to(DEVICE, non_blocking=True)

                with amp.autocast("cuda", dtype=amp_dtype, enabled=(DEVICE == "cuda")):
                    logits = tta_logits(ema_model, x)
                    v = crit_val(logits, y)

                val_loss += float(v.item()) * x.size(0)
                n_val += x.size(0)

                p = torch.sigmoid(logits).squeeze(1).float().cpu().numpy()
                probs.extend(p.tolist())
                ys.extend(y.squeeze(1).cpu().numpy().astype(int).tolist())

        val_loss /= max(1, n_val)
        probs = np.array(probs)
        ys = np.array(ys)

        thr, f1 = find_best_threshold(probs, ys)
        lr_now = float(opt.param_groups[0]["lr"])

        pd.DataFrame([{
            "epoch": ep,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_macroF1": f1,
            "thr": thr,
            "lr": lr_now,
        }]).to_csv(metrics_path, mode="a", header=False, index=False)

        sch.step()

        improved = ""
        if f1 > best_f1:
            best_f1, best_thr, best_ep = f1, thr, ep
            meta = {
                "best_thr": float(best_thr),
                "macro_f1": float(best_f1),
                "epoch": ep,
                "alpha": float(alpha),
                "model": MODEL_NAME,
                "img_size": IMG_SIZE,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            }
            save_best_model(ema_model, meta, OUT_DIR)
            epochs_no_improve = 0
            improved = "✅"
        else:
            epochs_no_improve += 1

        print(
            f"{improved} EP{ep:02d} | train_loss={tr_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_macroF1={f1:.4f} (thr={thr:.3f}) "
            f"| best={best_f1:.4f}@{best_ep} | lr={lr_now:.2e}"
        )

        save_last(model, ema_model, opt, ep, global_step, OUT_DIR)

        if epochs_no_improve >= PATIENCE:
            print(f"⛔ EarlyStopping: {PATIENCE} epochs no improvement (best F1={best_f1:.4f}@{best_ep})")
            break

    df = pd.read_csv(metrics_path)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 7))
    plt.subplot(3, 1, 1)
    plt.plot(df["epoch"], df["train_loss"], label="train")
    plt.plot(df["epoch"], df["val_loss"], label="val")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(df["val_macroF1"], label="F1")
    if best_ep != -1:
        plt.axhline(best_f1, linestyle="--", label=f"best={best_f1:.4f}@{best_ep}")
    plt.ylabel("Macro-F1")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(df["thr"], label="thr")
    plt.plot(df["lr"], label="lr")
    plt.xlabel("Epoch")
    plt.ylabel("thr / lr")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "curves.png"), dpi=150)
    print("[done] training finished, curves saved.")


if __name__ == "__main__":
    main()

