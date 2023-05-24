# coding=utf-8
"""
@Author : wangkangli
@Email: 455389059@qq.com
@createtime : 2023-05-24 0:15
"""

from config import *
from ./utils.func import normalize_std,apply_diff_freq
from ./utils.aug import mixup,core_mixup,spec_augmenter

import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchaudio



device="cuda" if torch.cuda.is_available() else "cpu"
ComputeDeltas = torchaudio.transforms.ComputeDeltas(win_length= 5)

def train(dataloader, model, loss_fn, optimizer, t,diff_freq_power,devices_no,scheduler):
    conf = config()
    size = len(dataloader.dataset)
    train_loss = 0
    n_train = 0
    correct = 0
    model.train()

    for batch, (X, y, y_3class) in enumerate(dataloader):

        if conf.MIXUP:
            X, y, y_3class = mixup(X, y, y_3class, t)
        if conf.DIFF_FREQ:
            X = apply_diff_freq(X, diff_freq_power, devices_no)

        X = normalize_std(X)
        X2 = ComputeDeltas(X)
        X2 = normalize_std(X2)
        X = torch.cat((X, X2), 1)

        if conf.SPEC_AUG:
            X = spec_augmenter(X)
        X, y, y_3class = X.to(device), y.to(device), y_3class.to(device)

        # Compute prediction error
        pred = model(X)
        loss, loss_3class = loss_fn(pred[:, :-3], y), loss_fn(pred[:, -3:], y_3class)

        # Backpropagation
        optimizer.zero_grad()
        optimizer.pc_backward([loss, loss_3class])
        optimizer.step()
        scheduler.step(t + 1)
        train_loss += loss.item()

        _, predicted = torch.max(pred[:, :-3].detach(), 1)
        _, y_predicted = torch.max(y.detach(), 1)
        correct += (predicted == y_predicted).sum().item()

        n_train += len(X)
        if batch % 500 == 0:
            loss_current, acc_current, current = train_loss / n_train, correct / n_train, batch * len(X)
            print(
                f"Train Epoch: {t + 1} loss: {loss_current:>7f}  accuracy: {acc_current:>7f} [{current:>5d}/{size:>5d}]")

    loss_current, acc_current = train_loss / n_train, correct / n_train
    return loss_current, acc_current


def val(dataloader, model, loss_fn,t):
    size = len(dataloader.dataset)
    val_loss = 0
    n_val = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for batch, (X, y, y_3class) in enumerate(dataloader):

            X, y, y_3class = X.to(device), y.to(device), y_3class.to(device)

            pred = model(X)
            loss, loss_3class = loss_fn(pred[:, :-3], y), loss_fn(pred[:, -3:], y_3class)

            val_loss += loss.item()

            _, predicted = torch.max(pred[:, :-3].detach(), 1)
            _, y_predicted = torch.max(y.detach(), 1)
            correct += (predicted == y_predicted).sum().item()

            n_val += len(X)
            if batch % 500 == 0:
                loss_current, acc_current, current = val_loss / n_val, correct / n_val, batch * len(X)
                print(
                    f"Val Epoch: {t + 1} loss: {loss_current:>7f}  accuracy: {acc_current:>7f} [{current:>5d}/{size:>5d}]")

    loss_current, acc_current = val_loss / n_val, correct / n_val
    return loss_current, acc_current
