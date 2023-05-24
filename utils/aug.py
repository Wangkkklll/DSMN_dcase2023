# coding=utf-8
"""
@Author : wangkangli
@Email: 455389059@qq.com
@createtime : 2023-05-24 0:12
"""
import torch
from torchlibrosa.augmentation import SpecAugmentation
import numpy as np


def core_mixup(X, y, ym, alpha=0.2, beta=0.2):
    indices = torch.randperm(X.size(0))
    X2 = X[indices, :, :, :]
    y2 = y[indices, :]
    ym2 = ym[indices, :]

    lam = torch.FloatTensor([np.random.beta(alpha, beta)])  

    X = lam * X + (1 - lam) * X2
    y = lam * y + (1 - lam) * y2
    ym = lam * ym + (1 - lam) * ym2

    return X, y, ym


def mixup(X, y, ym, epoch):
    if epoch < 60:
        X, y, ym = core_mixup(X, y, ym)
    else:
        X[:len(X) // 2, :, :, :], y[:len(y) // 2, :], ym[:len(ym) // 2, :] = core_mixup(X[:len(X) // 2, :, :, :],
                                                                                        y[:len(y) // 2, :],
                                                                                        ym[:len(ym) // 2, :])

    return X, y, ym


# SpecAugment
spec_augmenter = SpecAugmentation(
    time_drop_width=2,
    time_stripes_num=2,
    freq_drop_width=2,
    freq_stripes_num=2)
