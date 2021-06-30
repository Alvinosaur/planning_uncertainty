import numpy as np
import os
import torch
import sys
import re
import csv
import wandb\import datetime

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils import data
import matplotlib.pyplot as plt
import time

cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if cuda else "cpu")
if cuda:
    print("CUDA GPU!")
else:
    print("CPU!")


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform(m.weight)
        m.bias.data.fill_(0.01)


class Model(nn.Module):

    def __init__(self, input_size, interm_layer_sizes, output_size):
        super().__init__()
        modules = []
        modules.append(nn.Linear(input_size, interm_layer_sizes[0]))
        modules.append(nn.BatchNorm1d(interm_layer_sizes[0]))
        modules.append(nn.ReLU)

        for i in range(1, len(interm_layer_sizes)):
            modules.append(nn.Linear(interm_layer_sizes[i], interm_layer_sizes[i + 1]))
            modules.append(nn.BatchNorm1d(interm_layer_sizes[i + 1]))
            modules.append(nn.ReLU)

        modules.append(nn.Linear(interm_layer_sizes[-1], output_size))
        self.model = nn.Sequential(*modules).double()
        self.model.to(DEVICE)
        self.model.apply(init_weights)

    def forward(self, input):
        return self.model(input)


def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0
    for bi, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()

        x = x.to(DEVICE)
        y = y.to(DEVICE)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        del x
        del y

        running_loss += loss.item()

    return running_loss / len(train_loader)


def eval_model(model, loader, criterion):
    model.eval()
    running_loss = 0
    for bi, (x, y) in enumerate(loader):
        optimizer.zero_grad()

        x = x.to(DEVICE)
        y = y.to(DEVICE)
        pred = model(x)
        loss = criterion(pred, y)

        del x
        del y

        running_loss += loss.item()

    return running_loss / len(loader)


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, path):
    train_loss_series = []
    val_loss_series = []
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss = eval(model, val_loader, criterion)

        scheduler.step(metrics=val_loss)
        unique_name = "epoch_%d.h5" % epoch
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss}
        torch.save(checkpoint, os.path.join(path, unique_name))
        train_loss_series.append(train_loss)
        val_loss_series.append(val_loss)

    return train_loss_series, val_loss_series


class Dataset(data.Dataset):
    def __init__(self, data_files, batch_size):
        self.batch_size = batch_size
        self.X = []
        self.Y = []
        for f in data_files:
            data = np.load(f, allow_pickle=True)
            self.X.append(np.hstack([data["states"], data["actions"]]))
            self.Y.append([data["values"]])

        self.X = np.vstack(self.X)
        self.Y = np.vstack(self.Y)

        self.data_length = self.X.shape[0]

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        sample = self.X[index]
        label = self.Y[index]
        return torch.Tensor(sample), torch.Tensor(label)


if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 64 if cuda else 16
    num_epochs = 10
    USE_KAIMING_INIT = True
    date_time_str = '@'.join(str(datetime.datetime.now()).split(' '))
    PROGRESS_FOLDER = "/home/alvin/research/planning_uncertainty/learned_model/%s" % date_time_str
    lr = 1e-3
    wd = 5e-6
    num_workers = 8 if cuda else 0

    # Load data
    np.random.seed(12345)
    data_root = "/home/alvin/research/planning_uncertainty/beta"
    directories = os.listdir(data_root)
    all_files = []
    for dir in directories:
        files = os.listdir(os.path.join(data_root, dir))
        for fname in files:
            all_files.append(os.path.join(data_root, dir, fname))

    # randomly assign files to train or test
    all_files = np.random.permutation(all_files)
    num_train = len(all_files) * 0.8
    num_test = len(all_files) * 0.1
    train_files = all_files[:num_train]
    test_files = all_files[num_train:num_train + num_test]
    val_files = all_files[num_train + num_test:]

    train_dataset = Dataset(data_files=train_files,
                            batch_size=BATCH_SIZE)
    test_dataset = Dataset(data_files=test_files,
                           batch_size=BATCH_SIZE)
    val_dataset = Dataset(data_files=val_files,
                          batch_size=BATCH_SIZE)

    # optionally load from some previous checkpoint
    state_size = 3 + 7 + 7 + 1  # bottle pos + joint angles + joint actions + action length
    interm_layer_sizes = [64, 64, 16]
    model = Model(input_size=state_size, interm_layer_sizes=interm_layer_sizes, output_size=2)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    model_path = None
    if model_path is not None:
        checkpoint = torch.load(model_path)
        try:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint["scheduler"])
            print(checkpoint["val_loss"])
        except Exception as e:
            print(
                "Failed to load model, optimizer, and scheduler, defaulting to model because %s" % e)
            model.load_state_dict(checkpoint)
    else:
        if USE_KAIMING_INIT:
            model.apply(init_weights)

    if cuda:
        train_loader_args = dict(shuffle=True, batch_size=train_dataset.batch_size,
                                 num_workers=num_workers, pin_memory=True)
        test_loader_args = dict(shuffle=False, batch_size=test_dataset.batch_size,
                                num_workers=num_workers, pin_memory=True)
    else:
        train_loader_args = dict(shuffle=True, batch_size=train_dataset.batch_size)
        test_loader_args = dict(shuffle=False, batch_size=test_dataset.batch_size)

    train_loader = data.DataLoader(train_dataset, **train_loader_args)
    val_loader = data.DataLoader(val_dataset, **test_loader_args)
    test_loader = data.DataLoader(test_dataset, **test_loader_args)

    train_loss_series, val_loss_series = (
        train_model(model, train_loader=train_loader, val_loader=val_loader,
                    criterion=torch.nn.MSELoss, optimizer=optimizer,
                    path=PROGRESS_FOLDER, epochs=num_epochs))

    plt.plot(train_loss_series, label="train loss")
    plt.plot(val_loss_series, label="val loss")
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss vs epoch")
