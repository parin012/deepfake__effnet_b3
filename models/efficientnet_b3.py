#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import copy

import torch
import torch.nn as nn
import timm

from configs.config import MODEL_NAME, EMA_DECAY, GAMMA


def build_model():
    try:
        model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1)
    except Exception:
        print("[WARN] pretrained weights unavailable. Using random init.")
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1)
    return model


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma: float = GAMMA):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        w = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (w * (1 - pt).pow(self.gamma) * bce).mean()


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad = False
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k, v in esd.items():
            if k not in msd:
                continue
            src = msd[k]
            if v.dtype.is_floating_point:
                esd[k].copy_(v * self.decay + src * (1.0 - self.decay))
            else:
                esd[k].copy_(src)

