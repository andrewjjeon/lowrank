import torch
import torchvision
import matplotlib.pyplot as plt

import os
import yaml
import numpy as np
import pdb

from lfads import LFADS_Net
from utils import read_data, load_parameters, save_parameters, batchify_random_sample
from sklearn.decomposition import PCA
import mat73

# Select device to train LFADS on
device = 'cuda' if torch.cuda.is_available() else 'cpu'; print('Using device: %s'%device)

data = []

T_init = 4

# trial data
data = np.load('../data/andrew/sample_photostim_0113.npy', allow_pickle = True).item()
y = data['y']
u = data['u']
num_trials = y.shape[0]
num_steps = y.shape[1]
num_neurons = y.shape[2]
T_before = 4
T_after_start = 9
T_after_end = 13
y_before = np.mean(y[:,:T_before,:],axis = 1)
y_after = np.mean(y[:,T_after_start:T_after_end,:],axis = 1)

y_session = data['y_session']
u_session = data['u_session']

# linear interpolation of data
y_session_interp = y_session.copy()
for i in range(y_session.shape[1]):
    nan_start = -1
    nan_stop = -1
    for j in range(y_session.shape[0]):
        if nan_start == -1 and np.isnan(y_session_interp[j,i]):
            nan_start = j - 1
        if nan_start != -1 and not np.isnan(y_session_interp[j,i]):
            nan_stop = j
        if nan_start != -1 and nan_stop != -1:
            slope = y_session_interp[nan_stop,i] - y_session_interp[nan_start,i]
            for k in range(nan_stop - nan_start - 1):
                y_session_interp[nan_start + k + 1,i] = slope*k/(nan_stop-nan_start-1) + y_session_interp[nan_start,i]
            nan_start = -1
            nan_stop = -1
            
patterns = []
pattern_count = []
pattern_idx = []
pattern_length = []

start = -10
for t in range(u_session.shape[0]):
    if np.sum(np.abs(u_session[t,:])) > 0 and t > start + 4:
        idx = np.linspace(0,663-1,663).astype(int)
        on = u_session[t,:] > 0
        pattern = np.array(idx[on])
        start = t
        found = False
        for i in range(len(patterns)):
            if len(pattern) == len(patterns[i]):
                if np.linalg.norm(pattern - patterns[i]) == 0:
                    pattern_count[i] += 1
                    pattern_idx[i].append(t)
                    found = True
                    break
        if found is False:
            patterns.append(pattern)
            pattern_count.append(1)
            pattern_idx.append([t])
            pattern_length.append(len(pattern))

neuron_pattern = np.zeros(663)
for i in range(663):
    for p in patterns:
        if i in p:
            neuron_pattern[i] += 1
            
# remove patterns by neuron
'''
num_neurons = 5
removed_patterns = []
excluded_patterns = []
removed_neurons = []
removed_steps = np.zeros(u_session.shape[0])
removed_count = 0

while len(removed_neurons) < num_neurons:
    neuron = np.random.randint(0,663)
    if neuron not in removed_neurons and neuron_pattern[neuron] > 1:
        removed_neurons.append(neuron)
        removed = False
        for p_idx in range(len(patterns)):
            if neuron in patterns[p_idx] and p_idx not in removed_patterns and p_idx not in excluded_patterns and removed is False:
                removed_patterns.append(p_idx)
                removed_count += pattern_count[p_idx]
                for i in pattern_idx[p_idx]:
                    min_idx = np.max([0,i-20])
                    max_idx = np.min([i+50,u_session.shape[0]-1])
                    removed_steps[min_idx:max_idx] = 1
                removed = True
            if neuron in patterns[p_idx] and p_idx not in removed_patterns and p_idx not in excluded_patterns and removed is True:
                excluded_patterns.append(p_idx)
            
                
print(removed_count, np.sum(removed_steps) / u_session.shape[0])
print(len(removed_patterns))

test_indices = removed_steps.astype(int)
train_indices = np.ones(test_indices.shape[0]).astype(int) - test_indices.copy()
'''

# remove patterns randomly

num_patterns = 5
removed_patterns = []
removed_neurons = []
removed_steps = np.zeros(u_session.shape[0])
removed_count = 0

# reproducibility
np.random.seed(0)

while len(removed_patterns) < num_patterns:
    p_idx = np.random.randint(0,len(patterns))
    print('pattern id:', p_idx)
    if p_idx not in removed_patterns:
        removed_patterns.append(p_idx)
        removed_count += pattern_count[p_idx]
        removed_neurons.extend(patterns[p_idx])
        for i in pattern_idx[p_idx]:
            min_idx = np.max([0,i-20])
            max_idx = np.min([i+50,u_session.shape[0]-1])
            removed_steps[min_idx:max_idx] = 1
                
print(removed_count, np.sum(removed_steps) / u_session.shape[0])
print(len(removed_patterns))

test_indices = removed_steps.astype(int)
train_indices = np.ones(test_indices.shape[0]).astype(int) - test_indices.copy()
removed_neurons = list(set(removed_neurons))

train_test_split = {}
train_test_split['train_indices'] = train_indices
train_test_split['test_indices'] = test_indices
# np.save('train_test_split',train_test_split)

### data normalization
normalize = 1000
'''
y_session_interp_mean = np.nanmean(y_session_interp, axis = 0)
y_session_interp_std = np.nanmean(y_session_interp, axis = 0)
y_session_interp = (y_session_interp - y_session_interp_mean[np.newaxis, :])/y_session_interp_std[np.newaxis, :]
'''
hyperparams = load_parameters('./parameters.yaml')
save_parameters(hyperparams)
print(hyperparams)

time_window = hyperparams['time_window']

#### data preparation
F_train = []
input_train = []
input_id_train = []
for t in range(train_indices.shape[0]-time_window):
    if np.sum(train_indices[t:t+time_window+1]) == time_window + 1 and t > 5 + time_window:
        F_train.append(y_session_interp[t:t+time_window,:])
        input_id_train.append(u_session[t:t+time_window,:])
        input_train.append(
            np.repeat(np.expand_dims(
            np.max(u_session[t:t+time_window,:], axis = 1), axis = 1), 
            u_session[t:t+time_window,:].shape[1], axis = 1))
        
F_valid = []
input_valid = []
input_id_valid = []
for t in range(test_indices.shape[0]-time_window):
    if np.sum(test_indices[t:t+time_window+1]) == time_window + 1:
        F_valid.append(y_session_interp[t:t+time_window,:])
        input_id_valid.append(u_session[t:t+time_window,:])
        input_valid.append(
            np.repeat(np.expand_dims(
            np.max(u_session[t:t+time_window,:], axis = 1), axis = 1), 
            u_session[t:t+time_window,:].shape[1], axis = 1))

F_train_ts = torch.Tensor(np.stack(F_train)/normalize).to(device)
input_train_ts = torch.Tensor(np.stack(input_train)).to(device)
input_id_train_ts = torch.Tensor(np.stack(input_id_train)).to(device)

F_valid_ts = torch.Tensor(np.stack(F_valid)/normalize).to(device)
input_valid_ts = torch.Tensor(np.stack(input_valid)).to(device)
input_id_valid_ts = torch.Tensor(np.stack(input_id_valid)).to(device)

train_ds = torch.utils.data.TensorDataset(F_train_ts, torch.zeros_like(input_train_ts), input_id_train_ts)
valid_ds = torch.utils.data.TensorDataset(F_valid_ts, torch.zeros_like(input_valid_ts), input_id_valid_ts)

print(F_train_ts.shape)
print(F_valid_ts.shape)

num_trials, num_steps, num_cells = F_train_ts.shape

model = LFADS_Net(inputs_dim = num_cells,
                  T = num_steps, 
                  T_init = T_init, 
                  dt = 0.05,
                  device = device, 
                  model_hyperparams = hyperparams).to(device)

model.fit(train_ds, valid_ds, max_epochs=hyperparams['max_epoch'], batch_size=hyperparams['batch_size'], use_tensorboard=False)