# coding=utf-8
"""
@Author : wangkangli
@Email: 455389059@qq.com
@createtime : 2023-05-23 23:58
"""

from ./config import * #Set the configuration file name

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from tqdm import tqdm

conf = config()

spectrogram = T.MelSpectrogram(
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

def sc_data(root_path):

    setup_path = root_path + "evaluation_setup/"

    train_csv = pd.read_table(setup_path + "fold1_train.csv")

    devices_train_list = ["a","b","c","s1","s2","s3", ]
    devices_list = []
    for filename in tqdm(train_csv["filename"]):
        for device_train in devices_train_list:
            if re.search(device_train + ".wav", filename):
                devices_list.append(device_train)
    train_csv["device_name"] = devices_list
    train_csv.to_csv("fold1_train_add_device_name.csv")
    n_mels = conf.n_mels
    freq_bim = list(np.arange(0, n_mels, 1))
    a_file_list = train_csv[train_csv["device_name"] == "a"]
    bcs_file_list = train_csv[train_csv["device_name"] != "a"]
    dif_filename_list = []
    dif_freq_list = []
    dif_devive_list = []

    for filename_a in tqdm(a_file_list["filename"]):
        wave, _ = torchaudio.load(root_path + filename_a)

        spec_log_a = torch.log10(torch.mean(spectrogram(wave)[0, :, :], dim=1) + 1e-6)

        filename_bcs = bcs_file_list[bcs_file_list["filename"].str.contains(filename_a[:-5])]

        if len(filename_bcs) >= 1:
            for filename_temp, device_temp in zip(filename_bcs["filename"], filename_bcs["device_name"]):
                dif_filename_list.append(filename_temp)
                wave, _ = torchaudio.load(root_path + filename_temp)

                spec_log_temp = torch.log10(torch.mean(spectrogram(wave)[0, :, :], dim=1) + 1e-10)

                dif_freq_temp = spec_log_temp - spec_log_a
                dif_freq_list.append(dif_freq_temp.detach().numpy().copy())
                dif_devive_list.append(device_temp)
    df = pd.DataFrame(dif_filename_list, columns=["filename"])
    df["device_name"] = dif_devive_list
    df["dif_freq"] = dif_freq_list
    df.to_pickle(f"pro_data/{conf.process_data_f}/sc.pkl")
    dif_devices_train_list = ["b","c","s1","s2","s3", ]
    train_csv = pd.read_table(setup_path + "fold1_evaluate.csv")

    devices_train_list = ["a","b","c","s1","s2","s3","s4","s5","s6",]
    devices_list = []
    for filename in tqdm(train_csv["filename"]):
        for device_train in devices_train_list:
            if re.search(device_train + ".wav", filename):
                devices_list.append(device_train)

    train_csv["device_name"] = devices_list
    train_csv.to_csv("fold1_train_add_device_name_val.csv")
    a_file_list = train_csv[train_csv["device_name"] == "a"]
    bcs_file_list = train_csv[train_csv["device_name"] != "a"]
    dif_filename_list = []
    dif_freq_list = []
    dif_devive_list = []
    for filename_a in tqdm(a_file_list["filename"]):
        wave, _ = torchaudio.load(root_path + filename_a)
        spec_log_a = torch.log10(torch.mean(spectrogram(wave)[0, :, :], dim=1) + 1e-6)

        filename_bcs = bcs_file_list[bcs_file_list["filename"].str.contains(filename_a[:-5])]

        if len(filename_bcs) >= 1:
            for filename_temp, device_temp in zip(filename_bcs["filename"], filename_bcs["device_name"]):
                dif_filename_list.append(filename_temp)
                wave, _ = torchaudio.load(root_path + filename_temp)
                spec_log_temp = torch.log10(torch.mean(spectrogram(wave)[0, :, :], dim=1) + 1e-6)

                dif_freq_temp = spec_log_temp - spec_log_a
                dif_freq_list.append(dif_freq_temp.detach().numpy().copy())
                dif_devive_list.append(device_temp)
    df = pd.DataFrame(dif_filename_list, columns=["filename"])
    df["device_name"] = dif_devive_list
    df["dif_freq"] = dif_freq_list
    df.to_pickle(f"pro_data/{conf.process_data_f}sc_val.pkl")






