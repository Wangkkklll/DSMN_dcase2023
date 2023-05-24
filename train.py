# coding=utf-8
"""
@Author : wangkangli
@Email: 455389059@qq.com
@createtime : 2023-05-23 20:09
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from models.model1 import Cnn
from config import *
import NeSsi.nessi as nessi
from pcgrad import PCGrad
import timm
from timm.scheduler import CosineLRScheduler
from models import model1
from torch.utils.data import DataLoader, Dataset
from utils.func import normalize_std,apply_diff_freq
from utils.func import train_data_process,val_data_process
from dsmn_train.training import train, val
from utils.sc import *
import copy
conf = config()






device="cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
root_path = "data/TAU-urban-acoustic-scenes-2022-mobile-development/"
setup_path = root_path + "evaluation_setup/"
train_csv = pd.read_table(setup_path + "fold1_train.csv")
val_csv = pd.read_table(setup_path + "fold1_evaluate.csv")
test_csv = pd.read_table(setup_path + "fold1_test.csv")


label_list = train_csv["scene_label"].unique()
devices = ([
        "a",
        "b",
        "c",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6"
    ])

model = Cnn()
model.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")

model.train()
model = torch.quantization.prepare_qat(model).to(device)
loss_fn = nn.CrossEntropyLoss(reduction="sum")
optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3)
scheduler = CosineLRScheduler(optimizer, t_initial=100, lr_min=1e-6, warmup_t=5, warmup_lr_init=5e-5, warmup_prefix=True)
optimizer = PCGrad(optimizer)


train_X,train_y,train_y_3class,train_devices = train_data_process(train_csv,root_path,label_list,devices)
val_X,val_y,val_y_3class,val_devices = val_data_process(val_csv,root_path,label_list,devices)

train_dataset = torch.utils.data.TensorDataset(train_X, train_y, train_y_3class)
val_dataset = torch.utils.data.TensorDataset(val_X, val_y, val_y_3class)
train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=2)







model = Cnn()
model.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")


model.train()
model = torch.quantization.prepare_qat(model).to(device)
loss_fn = nn.CrossEntropyLoss(reduction="sum")
optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3)
scheduler = CosineLRScheduler(optimizer, t_initial=100, lr_min=1e-6, warmup_t=5, warmup_lr_init=5e-5, warmup_prefix=True)
optimizer = PCGrad(optimizer)



import matplotlib.pyplot as plt

max_acc = 5
epochs = []
train_acc_list = []
val_acc_list = []
train_loss_list = []
val_loss_list = []
for t in range(conf.epochs):
    # logs = {}
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer, t)
    val_loss, val_acc = val(val_dataloader, model, loss_fn,t)

    epochs.append(t)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)

    plt.subplot(1, 2, 1)
    t1, = plt.plot(epochs, train_acc_list, 'b-', label='training')
    v1, = plt.plot(epochs, val_acc_list, 'g-', label='validation')
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.legend(handles=[t1, v1], labels=['training', 'validation'])
    plt.subplot(1, 2, 2)
    t2, = plt.plot(epochs, train_loss_list, 'b-', label='training')
    v2, = plt.plot(epochs, val_loss_list, 'g-', label='validation')
    plt.xlabel('log loss')
    plt.ylabel('loss')
    plt.legend(handles=[t2, v2], labels=['training', 'validation'])
    plt.savefig("accuracy_loss.jpg")
    print("train_acc:", train_acc)
    print("val_acc:", val_acc)
    print("train_loss:", train_loss)
    print("val_loss:", val_loss)

    if max_acc < val_acc:
        max_acc = val_acc
        torch.save(copy.deepcopy(model).state_dict(), "model/model_best.pt")

torch.save(copy.deepcopy(model).state_dict(), "model/model_last.pt")


