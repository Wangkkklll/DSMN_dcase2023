# coding=utf-8
"""
@Author : wangkangli
@Email: 455389059@qq.com
@createtime : 2023-05-23 20:13
"""
from torch import nn
import torch
from ./config import *
import torchaudio
import torchaudio.transforms as T
from torchinfo import summary
from torchlibrosa.augmentation import SpecAugmentation
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
from tqdm import tqdm
import random

conf = config()
mel_spectrogram = T.MelSpectrogram(
    sample_rate=conf.sample_rate,
    n_fft=conf.n_fft,
    win_length=conf.win_length,
    hop_length=conf.hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney",
    onesided=True,
    n_mels=conf.n_mels,
    mel_scale="htk",
)

def wav_to_mel_log10(filepath):
    wave, _ = torchaudio.load(filepath)
    return torch.log10(mel_spectrogram(wave) + 1e-10)


def normalize_std(melspec):
    return (melspec - torch.mean(melspec, dim=(2, 3), keepdim=True)) / torch.std(melspec, dim=(2, 3), keepdim=True)


def label_to_onehot(scene_label, label_list):
    label_temp = torch.zeros(label_list.shape)
    label_temp[label_list == scene_label] = 1
    return label_temp


def get_devices_no(filename, devices):
    return devices.index(filename.split('-')[-1][:-4])


def label_for_multi(y):
    multi_y = np.zeros((y.shape[0], 3))
    for i in range(y.shape[0]):
        if np.argmax(y[i, :]) == 0 or np.argmax(y[i, :]) == 3 or np.argmax(y[i, :]) == 6:  # Indoor
            multi_y[i, 0] = 1
        elif np.argmax(y[i, :]) == 4 or np.argmax(y[i, :]) == 5 or np.argmax(y[i, :]) == 7 or np.argmax(
                y[i, :]) == 8:  # Outdoor
            multi_y[i, 1] = 1
        else:
            multi_y[i, 2] = 1  # Transportation
    return multi_y


def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow((spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


def plot_confusion_matrix(true, predicted,label_list):
    cm = confusion_matrix(true, predicted, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, square=True, cbar=False, annot=True, cmap="Blues")
    ax.set_xticklabels(label_list, rotation=90)
    ax.set_yticklabels(label_list, rotation=0)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")


def plot_device_wise_log_losses(loss_all, predicted_all, train_val_y, train_val_devices, devices,label_list):
    results_table = np.zeros((11, len(devices) + 2))

    for label_id, _ in enumerate(label_list):
        label_indx = (train_val_y[:, label_id] == 1)
        results_table[label_id, len(devices) + 1] = (predicted_all[
                                                         label_indx] == label_id).sum() / label_indx.sum() * 100
        results_table[label_id, 0] = loss_all[label_indx].mean()

        for device_id, _ in enumerate(devices):
            device_indx = np.array(train_val_devices) == device_id
            device_wise_indx = np.array(label_indx) * (device_indx)
            results_table[label_id, device_id + 1] = loss_all[device_wise_indx].mean()
            results_table[10, device_id + 1] = loss_all[device_indx].mean()

    results_table[10, len(devices) + 1] = (predicted_all == torch.argmax(train_val_y,
                                                                         dim=1).clone().numpy()).sum() / len(
        predicted_all) * 100
    results_table[10, 0] = loss_all.mean()

    df_results = pd.DataFrame(results_table, columns=["Log Loss", *devices, "Accuracy %"],
                              index=[*label_list, "Ovberall"])
    print(df_results)


def train_data_process(train_csv,root_path,label_list,devices):
    if conf.process_data:
        train_X = np.load(f"pro_data/{conf.process_data_f}train_X.npy")
        train_X = torch.from_numpy(train_X.astype(np.float32)).clone()
        train_y = np.load(f"pro_data/{conf.process_data_f}train_y.npy")
        train_y = torch.from_numpy(train_y.astype(np.float32)).clone()
        train_y_3class = np.load(f"pro_data/{conf.process_data_f}train_y_3class.npy")
        train_y_3class = torch.from_numpy(train_y_3class.astype(np.float32)).clone()
        train_devices = np.load(f"pro_data/{conf.process_data_f}train_devices.npy")
    else:
        train_X = []
        train_y = []
        train_devices = []

        for filename, scene_label in zip(tqdm(train_csv['filename']), train_csv['scene_label']):
            train_X.append(wav_to_mel_log10(root_path + filename))

            train_y.append(label_to_onehot(scene_label, label_list))
            train_devices.append(get_devices_no(filename, devices))

        train_X = torch.stack(train_X)
        train_y = torch.stack(train_y)

        train_y_3class = label_for_multi(train_y)
        train_y_3class = torch.from_numpy(train_y_3class.astype(np.float32)).clone()

        np.save(f"pro_data/{conf.process_data_f}train_X.npy", train_X)
        np.save(f"pro_data/{conf.process_data_f}train_y.npy", train_y)
        np.save(f"pro_data/{conf.process_data_f}train_y_3class.npy", train_y_3class)
        np.save(f"pro_data/{conf.process_data_f}train_devices.npy", train_devices)
    return train_X,train_y,train_y_3class,train_devices


def val_data_process(val_csv,root_path,label_list,devices):
    if conf.process_data:
        val_X = np.load(f"pro_data/{conf.process_data_f}val_X.npy")
        val_X = torch.from_numpy(val_X.astype(np.float32)).clone()
        val_y = np.load(f"pro_data/{conf.process_data_f}val_y.npy")
        val_y = torch.from_numpy(val_y.astype(np.float32)).clone()
        val_y_3class = np.load(f"pro_data/{conf.process_data_f}val_y_3class.npy")
        val_y_3class = torch.from_numpy(val_y_3class.astype(np.float32)).clone()
        val_devices = np.load(f"pro_data/{conf.process_data_f}val_devices.npy")
    else:
        val_X = []
        val_y = []
        val_devices = []

        for filename, scene_label in zip(tqdm(val_csv["filename"]), val_csv["scene_label"]):
            mel_spec = wav_to_mel_log10(root_path + filename)
            val_X.append(mel_spec)
            val_y.append(label_to_onehot(scene_label, label_list))
            val_devices.append(get_devices_no(filename, devices))

        val_X = torch.stack(val_X)
        val_y = torch.stack(val_y)

        val_y_3class = label_for_multi(val_y)
        val_y_3class = torch.from_numpy(val_y_3class.astype(np.float32)).clone()

        np.save(f"pro_data/{conf.process_data_f}val_X.npy", val_X)
        np.save(f"pro_data/{conf.process_data_f}val_y.npy", val_y)
        np.save(f"pro_data/{conf.process_data_f}val_y_3class.npy", val_y_3class)
        np.save(f"pro_data/{conf.process_data_f}val_devices.npy", val_devices)
    return val_X,val_y,val_y_3class,val_devices


def apply_diff_freq(X, diff_freq_power, devices_no):
    if random.randrange(0, 13, 1) != 0: # 1/13skip
        for idx, (X_temp, device_no) in enumerate(zip(X, devices_no)):
            tmp = (device_no==0)*diff_freq_power[random.randrange(0, len(diff_freq_power), 1),:].unsqueeze(0).unsqueeze(2)
            tmp = torch.from_numpy(np.dot(torch.ones((X.shape[3],1)), tmp[:,:,0])).clone()
            X[idx,0,:,:] = X_temp[0,:,:] + tmp.T

    return X






