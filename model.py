# model.py
import torch
import torch.nn as nn
from config import Config

class FocalLoss(nn.Module):
    """Focal Loss：自动降低简单样本权重，聚焦难样本"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        pt = pred * target + (1 - pred) * (1 - target)
        w = (1 - pt) ** self.gamma
        at = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (at * w * ce).mean()


class Tower(nn.Module):
    """双塔之一"""
    def __init__(self, dim, hidden, out):
        super().__init__()
        layers = []
        prev = dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        layers.append(nn.Linear(prev, out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return nn.functional.normalize(self.net(x), p=2, dim=1)


class DSSM(nn.Module):
    """DSSM 双塔模型"""
    def __init__(self, user_dim, item_dim, emb_dim=64):
        super().__init__()
        self.ut = Tower(user_dim, Config.HIDDEN, emb_dim)
        self.it = Tower(item_dim, Config.HIDDEN, emb_dim)

    def forward(self, u, i):
        return torch.sigmoid((self.ut(u) * self.it(i)).sum(dim=1))