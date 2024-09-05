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

learning_type = 'active'
print('learning_type', learning_type)
interp_len = 3
train_frac = 0.75
window_len = 16
train_ratio = 0.8
random_range = 10
T_init = 4
stim_dist = 10
num_unique_groups = 100
fixed_shift_index = 5
aug_times = 1
kl_factor = 10

if learning_type == 'active':
    hyperparams = load_parameters('./parameters_active_learning_simu_data_photostim')
    num_groups = int(hyperparams['num_groups'])
    kl_weight_schedule_dur = int(num_groups * kl_factor)
    kl_weight_schedule_start = int(num_groups * kl_factor / 2)
    hyperparams['kl_weight_schedule_dur'] = kl_weight_schedule_dur
    hyperparams['kl_weight_schedule_start'] = kl_weight_schedule_start
    l2_weight_schedule_dur = int(num_groups * kl_factor)
    hyperparams['l2_weight_schedule_dur'] = l2_weight_schedule_dur
    use_pca = hyperparams['use_pca']
    sparse = hyperparams['sparse']
    similarity = hyperparams['similarity']
    temp = hyperparams['temp']
    input_type = hyperparams['input_type']
    num_pc = hyperparams['num_pcs']
    num_active_groups = int(hyperparams['num_active_groups'])
    day = hyperparams['day']
    num_target_neurons = hyperparams['num_target_neurons']
    run_name = f"photostim_active_learning_g_1024_{day}_stim_group_{num_target_neurons}_num_groups_{num_groups}_num_active_groups_{num_active_groups}_sparse_{sparse}_similarity_{similarity}_temp_{temp}_input_type_{input_type}_day_1"
    hyperparams['run_name'] = run_name
    save_parameters(hyperparams)
    print(hyperparams)
    random_day = hyperparams['random_day']
    num_target_neurons = hyperparams['num_target_neurons']
    data_0 = np.load(f'../data/{random_day}_random_selection_holdout_{num_target_neurons}_sim_data.npy', allow_pickle=True).item()    
    train_num_groups = int(train_ratio * num_groups)
    F_train_vis_0 = data_0['F_data_train'][:train_num_groups]
    F_valid_vis_0 = data_0['F_data_train'][train_num_groups:num_groups]
    trial_stim_neuron_ids_train_0 = data_0['trial_stim_neuron_ids_train'][:train_num_groups]
    trial_stim_neuron_ids_valid_0 = data_0['trial_stim_neuron_ids_train'][train_num_groups:num_groups]
    data_name = f'{day}_active_selection_holdout_pca_{use_pca}_sparse_{sparse}_similarity_{similarity}_num_pcs_{num_pc}_temp_{temp}_input_type_{input_type}_sim_data.npy'
    print(data_name)
    data_1 = np.load('../data/' + data_name, allow_pickle=True).item()
    train_active_num_groups = int(train_ratio * num_active_groups)
    F_train_vis_1 = data_0['F_data_train'][:train_active_num_groups]
    F_valid_vis_1 = data_0['F_data_train'][train_active_num_groups:num_active_groups]
    trial_stim_neuron_ids_train_1 = data_1['trial_stim_neuron_ids_train'][:train_active_num_groups]
    trial_stim_neuron_ids_valid_1 = data_1['trial_stim_neuron_ids_train'][train_active_num_groups:num_active_groups]
    F_train_vis = np.concatenate((F_train_vis_0, F_train_vis_1))
    F_valid_vis = np.concatenate((F_valid_vis_0, F_valid_vis_1))
    trial_stim_neuron_ids_train = np.concatenate((trial_stim_neuron_ids_train_0, trial_stim_neuron_ids_train_1))
    trial_stim_neuron_ids_valid = np.concatenate((trial_stim_neuron_ids_valid_0, trial_stim_neuron_ids_valid_1))
else:
    hyperparams = load_parameters('./parameters_random_learning_simu_data_photostim.yaml')
    num_groups = int(hyperparams['num_groups'])
    kl_weight_schedule_dur = int(num_groups * kl_factor)
    kl_weight_schedule_start = int(num_groups * kl_factor / 2)
    hyperparams['kl_weight_schedule_dur'] = kl_weight_schedule_dur
    hyperparams['kl_weight_schedule_start'] = kl_weight_schedule_start
    l2_weight_schedule_dur = int(num_groups * kl_factor)
    hyperparams['l2_weight_schedule_dur'] = l2_weight_schedule_dur
    random_day = hyperparams['random_day']
    num_target_neurons = hyperparams['num_target_neurons']
    run_name = f"photostim_random_learning_g_1024_01_17_stim_group_{num_target_neurons}_num_groups_{num_groups}_rand_day_0"
    hyperparams['run_name'] = run_name
    save_parameters(hyperparams)
    print(hyperparams)
    data_1 = np.load(f'../data/{random_day}_random_selection_holdout_{num_target_neurons}_sim_data.npy', allow_pickle=True).item()    
    train_num_groups = int(train_ratio * num_groups)
    F_train_vis = data_1['F_data_train'][:train_num_groups]
    F_valid_vis = data_1['F_data_train'][train_num_groups:num_groups]
    trial_stim_neuron_ids_train = data_1['trial_stim_neuron_ids_train'][:train_num_groups]
    trial_stim_neuron_ids_valid = data_1['trial_stim_neuron_ids_train'][train_num_groups:num_groups]

num_train_trials = F_train_vis.shape[0]
num_neurons = F_train_vis.shape[2]
input_train_vis = np.zeros((num_train_trials, window_len, num_neurons))
input_train_vis[:,T_init+1:T_init+4,:] = 1
input_id_train_vis = (input_train_vis * np.repeat(trial_stim_neuron_ids_train, repeats = 16, axis = 1))

num_valid_trials = F_valid_vis.shape[0]
input_valid_vis = np.zeros((num_valid_trials, window_len, num_neurons))
input_valid_vis[:,T_init+1:T_init+4,:] = 1
input_id_valid_vis = (input_valid_vis * np.repeat(trial_stim_neuron_ids_valid, repeats = 16, axis = 1))

F_train_ts = torch.Tensor(F_train_vis).to(device)
input_train_ts = torch.Tensor(input_train_vis).to(device)
input_id_train_ts = torch.Tensor(input_id_train_vis).to(device)

F_valid_ts = torch.Tensor(F_valid_vis).to(device)
input_valid_ts = torch.Tensor(input_valid_vis).to(device)
input_id_valid_ts = torch.Tensor(input_id_valid_vis).to(device)

train_ds = torch.utils.data.TensorDataset(F_train_ts, input_train_ts, input_id_train_ts, input_id_train_ts, input_id_train_ts)
valid_ds = torch.utils.data.TensorDataset(F_valid_ts, input_valid_ts, input_id_valid_ts, input_id_valid_ts, input_id_valid_ts)

print(F_train_ts.shape)
print(F_valid_ts.shape)

num_trials, num_steps, num_cells = F_train_ts.shape

print('num_trials:', num_trials)
print('num_steps:', num_steps)
print('num_cells:', num_cells)

model = LFADS_Net(inputs_dim = num_cells, 
                  groups_dim = num_unique_groups, 
                  T = num_steps, 
                  T_init = T_init, 
                  dt = 0.05, 
                  device=device, 
                  model_hyperparams = hyperparams).to(device)

model.fit(train_ds, 
          valid_ds, 
          max_epochs = hyperparams['max_epoch'], 
          batch_size = hyperparams['batch_size'], 
          use_tensorboard = False)