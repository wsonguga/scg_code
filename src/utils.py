import numpy as np
import torch
import torch.nn as nn


def get_x_y(arr, num_labels):
    x = arr[:, :-num_labels]
    y = arr[:, -num_labels:]
    return x, y


def save_numpy_object(arr, path, name):
    with open(f"{path}/{name}.npy", "wb") as f:
        np.save(f, arr)


def load_numpy_object(path, name):
    with open(f"{path}/{name}.npy", "rb") as f:
        arr = np.load(f)
    return arr


def normalize(arr, path, load_values=False):
    if load_values is False:
        max_v = np.max(arr, axis=0).reshape(1, arr.shape[1])
        min_v = np.min(arr, axis=0).reshape(1, arr.shape[1])

        save_numpy_object(max_v, path, "normalize_max_values")
        save_numpy_object(min_v, path, "normalize_min_values")
    else:
        max_v = load_numpy_object(path, "normalize_max_values")
        min_v = load_numpy_object(path, "normalize_min_values")

    arr = np.divide(arr - min_v, max_v - min_v)

    return arr, min_v, max_v


def get_data(data_path, out_path, name, load_values, device, num_labels,
             return_extra, drop_last=False, drop_extra=0):
    arr = np.load(f"{data_path}/{name}.npy")

    if drop_extra != 0:
        arr = np.concatenate((arr[:, :-num_labels-drop_extra],
                             arr[:, -num_labels:]), axis=1)

    _, Y = get_x_y(arr, num_labels)

    if drop_last:
        arr = arr[:, :-1]

    arr, min_v, max_v = normalize(arr, out_path, load_values)
    arr_x, arr_y = get_x_y(arr, num_labels)

    arr_x = torch.from_numpy(arr_x).to(dtype=torch.float32, device=device)
    arr_y = torch.from_numpy(arr_y).to(dtype=torch.float32, device=device)

    Y = torch.from_numpy(Y).to(dtype=torch.float32, device=device)
    min_v = torch.from_numpy(min_v).to(dtype=torch.float32, device=device)
    max_v = torch.from_numpy(max_v).to(dtype=torch.float32, device=device)

    min_v = min_v[0, -num_labels:]
    max_v = max_v[0, -num_labels:]

    if return_extra:
        return arr_x, arr_y, Y, min_v, max_v
    else:
        return arr_x, arr_y


class Model(nn.Module):
    def __init__(self, input_size, num_labels, hidden_size=1024,
                 num_layers=4, p=0.5):
        super().__init__()
        layers = []
        for i in range(num_layers-1):
            in_size = input_size if i == 0 else hidden_size
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=p))
        layers.append(nn.Linear(hidden_size, num_labels))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


def train_model(model, optim, criterion, x, y, scheduler=None):
    model.train()
    optim.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optim.step()
    if scheduler is not None:
        try:
            scheduler.step()
        except TypeError:
            try:
                scheduler.step(loss)
            except TypeError:
                pass
    return loss.item()


@torch.no_grad()
def test_model(model, x, y, min_v, max_v):
    model.eval()
    out = model(x)
    out = out * (max_v - min_v) + min_v
    losses = torch.mean(torch.abs(out - y), axis=0)
    return losses


# %%
import numpy as np
import time
import sys
import re
import pytz
import time
import math
import subprocess
import sys
import random
import webbrowser
import numpy as np
from datetime import datetime
from dateutil import tz

def mac_to_int(mac):
    res = re.match('^((?:(?:[0-9a-f]{2}):){5}[0-9a-f]{2})$', mac.lower())
    if res is None:
        raise ValueError('invalid mac address')
    return int(res.group(0).replace(':', ''), 16)

def int_to_mac(macint):
    # if type(macint) != int:
    #     raise ValueError('invalid integer')
    newint = int(macint)
    return ':'.join(['{}{}'.format(a, b)
                     for a, b
                     in zip(*[iter('{:012x}'.format(newint))]*2)])

# This function converts the time string to epoch time xxx.xxx (second.ms).
# Example: time = "2020-08-13T02:03:00.200", zone = "UTC" or "America/New_York"
# If time = "2020-08-13T02:03:00.200Z" in UTC time, then call timestamp = local_time_epoch(time[:-1], "UTC"), which removes 'Z' in the string end
def local_time_epoch(time, zone):
    local_tz = pytz.timezone(zone)
    localTime = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f")
    local_dt = local_tz.localize(localTime, is_dst=None)
    # utc_dt = local_dt.astimezone(pytz.utc)
    epoch = local_dt.timestamp()
    # print("epoch time:", epoch) # this is the epoch time in seconds, times 1000 will become epoch time in milliseconds
    # print(type(epoch)) # float
    return epoch 

# This function converts the epoch time xxx.xxx (second.ms) to time string.
# Example: time = "2020-08-13T02:03:00.200", zone = "UTC" or "America/New_York"
# If time = "2020-08-13T02:03:00.200Z" in UTC time, then call timestamp = local_time_epoch(time[:-1], "UTC"), which removes 'Z' in the string end
def epoch_time_local(epoch, zone):
    local_tz = pytz.timezone(zone)
    time = datetime.fromtimestamp(epoch).astimezone(local_tz).strftime("%Y-%m-%dT%H:%M:%S.%f")
    return time 

def sort_dataset():
    timestamp_index = -5 # -5 timestamp
    mac_index = -6 # mac address

    file_path = "../data/real_regression_data/real_train_timesorted.npy"
    train_data = np.load(file_path)

    file_path = "../data/real_regression_data/real_test_timesorted.npy"
    test_data = np.load(file_path)

    # print(train_data.shape, np.min(train_data[:, timestamp_index]), np.max(train_data[:, timestamp_index]))
    # print(test_data.shape, np.min(test_data[:, timestamp_index]), np.max(test_data[:, timestamp_index]))
    print(train_data.shape, 
        epoch_time_local(np.min(train_data[:, timestamp_index])/10e8, "America/New_York"), 
        epoch_time_local(np.max(train_data[:, timestamp_index])/10e8, "America/New_York")
    )
    print(test_data.shape, 
        epoch_time_local(np.min(test_data[:, timestamp_index])/10e8, "America/New_York"), 
        epoch_time_local(np.max(test_data[:, timestamp_index])/10e8, "America/New_York")
    )

    data_set = np.concatenate( (train_data, test_data), 0)
    data_set = data_set[np.argsort(data_set[:, timestamp_index])]

    # mac = int_to_mac(data_set[0, mac_index])
    # print(mac)
    # mac = int_to_mac(data_set[1, mac_index])
    # print(mac)
    # mac = int_to_mac(data_set[-1, mac_index])
    # print(mac)

    # time = epoch_time_local(data_set[0, timestamp_index]/10e8, "America/New_York")
    # print(time)    
    # time = epoch_time_local(data_set[1, timestamp_index]/10e8, "America/New_York")
    # print(time)       
    # time = epoch_time_local(data_set[-1, timestamp_index]/10e8, "America/New_York")
    # print(time)   

    split_index = train_data.shape[0]
    train_data = data_set[:split_index, :]
    test_data = data_set[split_index:, :]

    print(train_data.shape, 
        epoch_time_local(np.min(train_data[:, timestamp_index])/10e8, "America/New_York"), 
        epoch_time_local(np.max(train_data[:, timestamp_index])/10e8, "America/New_York")
    )
    print(test_data.shape, 
        epoch_time_local(np.min(test_data[:, timestamp_index])/10e8, "America/New_York"), 
        epoch_time_local(np.max(test_data[:, timestamp_index])/10e8, "America/New_York")
    )
    # print(test_data.shape, np.min(test_data[:, timestamp_index]), np.max(test_data[:, timestamp_index]))

    # np.save("../data/real_regression_data/real_train_truesorted.npy", train_data)
    # np.save("../data/real_regression_data/real_test_truesorted.npy", test_data)
    train_data = np.load("../data/real_regression_data/real_train_truesorted.npy")
    test_data = np.load("../data/real_regression_data/real_test_truesorted.npy")

    # mac = int_to_mac(data_set[0, mac_index])
    # print(mac)
    # mac = int_to_mac(data_set[1, mac_index])
    # print(mac)
    # mac = int_to_mac(data_set[-1, mac_index])
    # print(mac)

    # time = epoch_time_local(data_set[0, timestamp_index]/10e8, "America/New_York")
    # print(time)    
    # time = epoch_time_local(data_set[1, timestamp_index]/10e8, "America/New_York")
    # print(time)       
    # time = epoch_time_local(data_set[-1, timestamp_index]/10e8, "America/New_York")
    # print(time)   

    print(train_data.shape, 
        epoch_time_local(np.min(train_data[:, timestamp_index])/10e8, "America/New_York"), 
        epoch_time_local(np.max(train_data[:, timestamp_index])/10e8, "America/New_York")
    )
    print(test_data.shape, 
        epoch_time_local(np.min(test_data[:, timestamp_index])/10e8, "America/New_York"), 
        epoch_time_local(np.max(test_data[:, timestamp_index])/10e8, "America/New_York")
    )

sort_dataset()
# %%
