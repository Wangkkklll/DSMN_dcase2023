from config_submit1 import * #Set the configuration file name
conf = config()
print(conf.epochs)

import copy
import random

from IPython.display import clear_output
from livelossplot import PlotLosses
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import timm
from timm.scheduler import CosineLRScheduler
import torch
from torch import nn
import torchaudio
import torchaudio.transforms as T
from torchinfo import summary
from torchlibrosa.augmentation import SpecAugmentation 
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

import NeSsi.nessi as nessi
from pcgrad import PCGrad




def train(dataloader, model, loss_fn, optimizer, t):
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
        X = torch.cat((X,X2),1)
        
        if conf.SPEC_AUG:
            X = spec_augmenter(X)
        X, y, y_3class = X.to(device), y.to(device), y_3class.to(device)

        # Compute prediction error
        pred = model(X)
        loss, loss_3class = loss_fn(pred[:,:-3], y), loss_fn(pred[:,-3:], y_3class)

        # Backpropagation
        optimizer.zero_grad()
        optimizer.pc_backward([loss, loss_3class]) 
        optimizer.step()
        scheduler.step(t+1)
        train_loss += loss.item()
        
        _, predicted = torch.max(pred[:,:-3].detach(), 1)
        _, y_predicted = torch.max(y.detach(), 1)
        correct += (predicted == y_predicted).sum().item()
        
        n_train += len(X)
        if batch % 500 == 0:
            loss_current, acc_current, current = train_loss/n_train, correct/n_train, batch * len(X)
            print(f"Train Epoch: {t+1} loss: {loss_current:>7f}  accuracy: {acc_current:>7f} [{current:>5d}/{size:>5d}]")
    
    loss_current, acc_current = train_loss/n_train, correct/n_train   
    return loss_current, acc_current


def val(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    val_loss = 0
    n_val = 0
    correct = 0
    model.eval()
    with torch.no_grad():        
        for batch, (X, y, y_3class) in enumerate(dataloader):

            X, y, y_3class = X.to(device), y.to(device), y_3class.to(device)
            
            pred = model(X)
            loss, loss_3class = loss_fn(pred[:,:-3], y), loss_fn(pred[:,-3:], y_3class)

            val_loss += loss.item()
            
            _, predicted = torch.max(pred[:,:-3].detach(), 1)
            _, y_predicted = torch.max(y.detach(), 1)
            correct += (predicted == y_predicted).sum().item()
            
            n_val += len(X)
            if batch % 500 == 0:
                loss_current, acc_current, current = val_loss/n_val, correct/n_val, batch*len(X)
                print(f"Val Epoch: {t+1} loss: {loss_current:>7f}  accuracy: {acc_current:>7f} [{current:>5d}/{size:>5d}]")
                
    loss_current, acc_current = val_loss/n_val, correct/n_val
    return loss_current, acc_current

liveloss = PlotLosses()
min_loss = 5
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
    return (melspec-torch.mean(melspec, dim=(2,3), keepdim=True)) / torch.std(melspec, dim=(2,3), keepdim=True)


def label_to_onehot(scene_label, label_list):
    label_temp = torch.zeros(label_list.shape)
    label_temp[label_list==scene_label] = 1
    return label_temp


def get_devices_no(filename, devices):
    return devices.index(filename.split('-')[-1][:-4])


def label_for_multi(y):
    multi_y = np.zeros((y.shape[0],3))
    for i in range(y.shape[0]):
        if np.argmax(y[i,:])==0 or np.argmax(y[i,:])==3 or np.argmax(y[i,:])==6: #Indoor
            multi_y[i,0] = 1
        elif np.argmax(y[i,:])==4 or np.argmax(y[i,:])==5 or np.argmax(y[i,:])==7 or np.argmax(y[i,:])==8: #Outdoor
            multi_y[i,1] = 1
        else:
            multi_y[i,2] = 1 #Transportation
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
    
    
def plot_confusion_matrix(true, predicted):
    cm = confusion_matrix(true, predicted, normalize="true")
    fig, ax = plt.subplots(figsize = (10,10))
    sns.heatmap(cm, square=True, cbar=False, annot=True, cmap="Blues")
    ax.set_xticklabels(label_list, rotation=90) 
    ax.set_yticklabels(label_list, rotation=0) 
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    
    
def plot_device_wise_log_losses(loss_all, predicted_all, train_val_y, train_val_devices, devices):
    results_table = np.zeros((11, len(devices)+2))

    for label_id, _ in enumerate(label_list):
        label_indx = (train_val_y[:,label_id]==1)
        results_table[label_id, len(devices)+1] = (predicted_all[label_indx]==label_id).sum()/label_indx.sum()*100
        results_table[label_id, 0] = loss_all[label_indx].mean()

        for device_id, _ in enumerate(devices):
            device_indx = np.array(train_val_devices)==device_id
            device_wise_indx = np.array(label_indx)*(device_indx)
            results_table[label_id, device_id + 1] = loss_all[device_wise_indx].mean()
            results_table[10, device_id+1] = loss_all[device_indx].mean()

    results_table[10, len(devices)+1] = (predicted_all == torch.argmax(train_val_y, dim=1).clone().numpy()).sum()/len(predicted_all)*100
    results_table[10, 0] = loss_all.mean()

    df_results = pd.DataFrame(results_table, columns=["Log Loss", *devices, "Accuracy %"], index=[ *label_list, "Ovberall"])    
    display(df_results)  
    
if True:
    train_X = np.load(f"reuse/{conf.reusefolder}train_X.npy")
    train_X = torch.from_numpy(train_X.astype(np.float32)).clone()
    train_y = np.load(f"reuse/{conf.reusefolder}train_y.npy")
    train_y = torch.from_numpy(train_y.astype(np.float32)).clone()
    train_y_3class = np.load(f"reuse/{conf.reusefolder}train_y_3class.npy")
    train_y_3class = torch.from_numpy(train_y_3class.astype(np.float32)).clone()
    train_devices = np.load(f"reuse/{conf.reusefolder}train_devices.npy")
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
    
    np.save(f"reuse/{conf.reusefolder}train_X.npy", train_X)
    np.save(f"reuse/{conf.reusefolder}train_y.npy", train_y)
    np.save(f"reuse/{conf.reusefolder}train_y_3class.npy", train_y_3class)
    np.save(f"reuse/{conf.reusefolder}train_devices.npy", train_devices)

#print(train_X.shape)

if True:
    val_X = np.load(f"reuse/{conf.reusefolder}val_X.npy")
    val_X = torch.from_numpy(val_X.astype(np.float32)).clone()
    val_y = np.load(f"reuse/{conf.reusefolder}val_y.npy")
    val_y = torch.from_numpy(val_y.astype(np.float32)).clone()
    val_y_3class = np.load(f"reuse/{conf.reusefolder}val_y_3class.npy")
    val_y_3class = torch.from_numpy(val_y_3class.astype(np.float32)).clone()
    val_devices = np.load(f"reuse/{conf.reusefolder}val_devices.npy")
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
    
    np.save(f"reuse/{conf.reusefolder}val_X.npy", val_X)
    np.save(f"reuse/{conf.reusefolder}val_y.npy", val_y)
    np.save(f"reuse/{conf.reusefolder}val_y_3class.npy", val_y_3class)
    np.save(f"reuse/{conf.reusefolder}val_devices.npy", val_devices)
    
if conf.include_val:
    train_X = torch.cat((train_X, val_X), 0)
    train_y = torch.cat((train_y, val_y), 0)
    train_y_3class = torch.cat((train_y_3class, val_y_3class), 0)
    train_devices = np.concatenate((train_devices, val_devices), 0)

val_X = normalize_std(val_X)

ComputeDeltas = torchaudio.transforms.ComputeDeltas(win_length= 5)
val_X2 = ComputeDeltas(val_X)
val_X2 = normalize_std(val_X2)
val_X = torch.cat((val_X,val_X2), 1)

# Create data loaders.
train_dataset = torch.utils.data.TensorDataset(train_X, train_y, train_y_3class)
val_dataset = torch.utils.data.TensorDataset(val_X, val_y, val_y_3class)
train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=2)
for X, y, y_3class in train_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
    
del train_X, train_y

#################################

n_output = label_list.shape[0] + 3
n_hidden = 100
import math
# import model
from models.model1 import Cnn

model = Cnn()
model.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
########################################



print(nessi.get_model_size(model, "torch", input_size = (1,2,X.shape[2],X.shape[3])))
model.train()
model = torch.quantization.prepare_qat(model).to(device)
loss_fn = nn.CrossEntropyLoss(reduction="sum")
optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3)
scheduler = CosineLRScheduler(optimizer, t_initial=100, lr_min=1e-6, warmup_t=5, warmup_lr_init=5e-5, warmup_prefix=True)
optimizer = PCGrad(optimizer)

def core_mixup(X, y, ym, alpha=0.2, beta=0.2):

    indices = torch.randperm(X.size(0)) 
    X2 = X[indices,:,:,:]
    y2 = y[indices,:]
    ym2 = ym[indices,:]

    lam = torch.FloatTensor([np.random.beta(alpha, beta)])  #

    X = lam*X + (1-lam)*X2  
    y = lam*y + (1-lam)*y2
    ym = lam*ym + (1-lam)*ym2
    
    return  X, y, ym


def mixup(X, y, ym, epoch):
    
    if epoch < 60:
        X, y, ym = core_mixup(X, y, ym)            
    else:
        X[:len(X)//2,:,:,:], y[:len(y)//2,:], ym[:len(ym)//2,:] = core_mixup(X[:len(X)//2,:,:,:], y[:len(y)//2,:], ym[:len(ym)//2,:])
            
    return X, y, ym

#SpecAugment
spec_augmenter = SpecAugmentation(
            time_drop_width=2,
            time_stripes_num=2,
            freq_drop_width=2,
            freq_stripes_num=2)

if conf.DIFF_FREQ:
    diff_freq_list = pd.read_pickle(f"reuse/{conf.reusefolder}diff_freq.pkl") 
    if conf.include_val:
        diff_freq_list_val = pd.read_pickle(f"reuse/{conf.reusefolder}diff_freq_val.pkl") 
        diff_freq_list = pd.concat([diff_freq_list, diff_freq_list_val])
    print((diff_freq_list.head()))

    dif_devices_train_list = [
            "b",
            "c",
            "s1",
            "s2",
            "s3",
        ]

    diff_freq_power = torch.from_numpy(np.stack(diff_freq_list["dif_freq"].values).astype(np.float32)).clone()
    
def apply_diff_freq(X, diff_freq_power, devices_no):
    if random.randrange(0, 13, 1) != 0: # 1/13skip
        for idx, (X_temp, device_no) in enumerate(zip(X, devices_no)):
            tmp = (device_no==0)*diff_freq_power[random.randrange(0, len(diff_freq_power), 1),:].unsqueeze(0).unsqueeze(2)
            tmp = torch.from_numpy(np.dot(torch.ones((X.shape[3],1)), tmp[:,:,0])).clone()
            X[idx,0,:,:] = X_temp[0,:,:] + tmp.T

    return X

#Check the effect
for X, y, devices_no in train_dataloader:
    X, y = X.to("cpu"), y.to("cpu")
    X_ori = normalize_std(X)
    plot_spectrogram(X_ori[0,0,:,:])
    X = apply_diff_freq(X, diff_freq_power, devices_no)
    X = normalize_std(X)
    plot_spectrogram((X[0,0,:,:]))
    print(devices_no[0])
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
    

import matplotlib.pyplot as plt
min_loss = 5
epochs = []
train_acc_list = []
val_acc_list = []
train_loss_list = []
val_loss_list = []
for t in range(conf.epochs):
    # logs = {}
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer, t)
    val_loss, val_acc = val(val_dataloader, model, loss_fn)
    
    epochs.append(t)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    
    plt.subplot(1,2,1)
    t1, = plt.plot(epochs, train_acc_list, 'b-', label='training')
    v1, = plt.plot(epochs, val_acc_list, 'g-', label='validation')
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.legend(handles=[t1, v1], labels=['training','validation'])
    plt.subplot(1, 2, 2)
    t2, = plt.plot(epochs, train_loss_list, 'b-', label='training')
    v2, = plt.plot(epochs, val_loss_list, 'g-', label='validation')
    plt.xlabel('log loss')
    plt.ylabel('loss')
    plt.legend(handles=[t2, v2], labels=['training','validation'])
    plt.savefig("accuracy_loss.jpg")
    print("train_acc:",train_acc)
    print("val_acc:",val_acc)
    print("train_loss:",train_loss)
    print("val_loss:",val_loss)
    
    
    if min_loss > val_loss:
        min_loss = val_loss
        torch.save(copy.deepcopy(model).state_dict(), "model/model_min.pt")

torch.save(copy.deepcopy(model).state_dict(), "model/model.pt")
