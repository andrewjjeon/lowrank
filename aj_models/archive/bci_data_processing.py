import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy.linalg as la
import os
import yaml
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
mpl.rcParams['figure.dpi'] = 80
# BCI DATA PROCESSING
# ------------------------------------------------------------------


# RAW DATA DESCRIPTION
# ------------------------------------------------------------------

# dt_si: time step in seconds
# iscell: 1 if cell is neuron, else 0
# dat_file: name of file
# session: date of session
# mouse: name of mouse
# mean_image: session mean image of the FOV
# fov: name of field of view
# dprime_1dFperF: the d-prime of each ROI for a 100% dF/F change
# version: version of this data format

# spont: (dictionary)
# - Ftrace: raw fluorescence
# - trace_corr: correlations of spontaneous activity

# BCI_1: (dictionary) first BCI task of day, BCI_2, BCI_3, etc.....
# - F: deltaF/F during behavior
# - Fraw: raw flueorescence intensity during behavior
# - df_closedloop: dF/F0 during behavior (F0 is numpy.std(Fraw))
# - centroidX: mean X pixel indices for each neuron
# - centroidY: mean Y pixel indices for each neuron
# - Ftrace: raw fluorescence
# - trace_corr: correlations of activity
# - dist: distance from conditioned neuron in pixels
# - conditioned_neuron_coordinates: location of CN in pixels (X,Y)
# - conditioned_neuron: index of conditioned neuron
# - reward_time: time of rewards
# - step_time: time of lickport steps
# - Trial_start: time of trial starts
# - lick_time: time of licks
# - threshold_crossing_time: time of threshold crossing

# photostim: (dictionary)
# - FstimRaw: (neurons x time) raw fluorescence intensity during photostimulation
# - Fstim: (time x neurons x photostimulation trials) dF/F0 of photostim experiments where F0 i sthe avg F before stimulation
# - seq: (array of int) photostimulation group IDs
# - favg: (time x neurons x photostim groups) avg response to photostimulation 
# - stimDist: (neurons x photostim groups) distance of each cell from closest photostimulation beamlet in pixels
# - stimPosition: (beamlet x 2 x photostim groups) XY coordinates of each photostim beamlet in each photostim group
# - centroidX: (list of neurons) mean X pixel indices for each neuron
# - centroidY: (list of neurons) mean Y pixel indices for each neuron
# - slmDist: (neurons x photostim groups) distance of each neuron from each center of the SLM in pixels
# - stimID: (array of int time) all 0s except at onset of photostimulations, where its value is photostim group index

os.chdir("C:/Users/andre/Desktop/active_neuron")
with open("aj_models/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

mouse = config['mouse']
date = config['date']

normalize = config['normalize']
ark_order = config['ark_order']
interp_len = config['ark_order'] - 1



try:
    data_all = np.load('data/BCI_' + mouse + '-' + date + '.npy', allow_pickle = True).item()
    print("Keys of raw data: ", data_all.keys())
except FileNotFoundError:
    print("This is not a valid mouse + date combination. Please try again.")

data = []
dataset_id = 0
data.append(data_all['photostim']) #This is a dictionary

print(f"The keys of the photostim dictionary are: {data[dataset_id].keys()}")
print(" ")
print(f"FstimRaw: {data_all['photostim']['FstimRaw'].shape} (neurons, time) raw fluorescence intensity during photostimulation")
print(f"Fstim: {data_all['photostim']['Fstim'].shape} (time, neurons, photostim trials) dF/F0 of photostim experiments where F0 i sthe avg F before stimulation")
print(f"seq: {len(data_all['photostim']['seq'])} array of int photostimulation trials")
print(f"favg: {data_all['photostim']['favg'].shape} (time, neurons, photostim groups) avg response to photostimulation")
print(f"stimDist: {data_all['photostim']['stimDist'].shape}  (neurons x photostim groups) distance of each cell from closest photostimulation beamlet in pixels")
print(f"stimPosition: {data_all['photostim']['stimPosition'].shape} (beamlet x 2 x photostim groups) XY coordinates of each photostim beamlet in each photostim group")
print(f"centroidX: {len(data_all['photostim']['centroidX'])} (list of neurons) mean X pixel indices for each neuron")
print(f"centroidY: {len(data_all['photostim']['centroidY'])} (list of neurons) mean Y pixel indices for each neuron")
print(f"slmDist: {data_all['photostim']['slmDist'].shape} (neurons x photostim groups) distance of each neuron from each center of the SLM in pixels")
print(f"stimID: {len(data_all['photostim']['stimID'])} (array of int time) all 0s except at onset of photostimulations, where its value is photostim group index")

seq = data[dataset_id]['seq']
stimDist = data[dataset_id]['stimDist']
stim_group_id = 0

#neurons with distance to stimulation laser of less than 50 in photostim group 0
close_perturbation_center_neuron_ids = data_all['photostim']['slmDist'][:,stim_group_id] < 50

#indices of neurons with a distance to stimulation target of less than 30 in photostim group 0
target_neuron_ids = np.where(data_all['photostim']['stimDist'][:,stim_group_id] < 30)[0]



plt.figure(figsize=(12, 6))
#all neurons
plt.scatter(data_all['photostim']['centroidX'], data_all['photostim']['centroidY'], s = 2, label = "All Neurons")

#orange is the stimulation points, coordinates of the 10 photostim beamlets for photostim group 0
plt.scatter(data[dataset_id]['stimPosition'][:,0,stim_group_id], data[dataset_id]['stimPosition'][:,1,stim_group_id], s = 6, alpha = 1.0, color = 'orange', label = 'Beamlets')

#target neurons with distance of <30 from beamlets
plt.scatter(np.asarray(data_all['photostim']['centroidX'])[target_neuron_ids], np.asarray(data_all['photostim']['centroidY'])[target_neuron_ids], color = 'green', s = 2, label = 'Target Neurons')
plt.title('Coordinates of Neurons, Beamlets, and Stimulated Neurons of Photostim Group 0')
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')
plt.legend()



### padding zero right end to correct the shift
# all element except last one = all element except first one
# stimID_correct[:-1] = data[dataset_id]['stimID'][1:]
stimID_correct = (data[dataset_id]['stimID']).copy()
print(f"{stimID_correct} stimID_correct is 0 at all t's except when there's a stimulus, there it's marked with the photostim group id")

plt.figure(figsize=(12, 6))
plt.plot(np.mean(data[dataset_id]['FstimRaw'],axis=0)[:50], label='mean fluorescence intensity of all neurons during photostim')
plt.xlabel("time: first 50")
plt.ylabel("fluorescence intensity for blue / photostim group index for orange")
plt.plot(stimID_correct[:50], label='photostim group index number')         
plt.legend()  

# stimID_correct is 0 at all 29308 t's except when there's a stimulus
# there it's marked with the photostim group id like [ 0. 74.  0. ...  0.  0.  0.]
num_trials = np.where(stimID_correct != 0)[0].shape[0]  # EX: num_trials=2046
num_neurons = data[dataset_id]['FstimRaw'].shape[0]  # EX: 502
Fstim_interp = data[dataset_id]['FstimRaw'].copy()
Fstim_baseline_mean = np.mean(Fstim_interp, axis = 1)  # Fstim_baseline_mean is the mean of 29308 intensity readings for each of the 502 neurons
Fstim_baseline_std = np.std(Fstim_interp, axis = 1)  # Fstim_baseline_std is the standard deviation of 29308 intensity readings for each of the 502 neurons
Fstim_interp_norm = Fstim_interp
stim_steps = np.where(stimID_correct != 0)[0]  # EX: (2046,)



stim_dist = 10
num_unique_groups = 100
stim_input = np.zeros(Fstim_interp.shape[1])  # EX: (29308,)
stim_input_id = np.zeros((Fstim_interp.shape[1], num_neurons))  # EX: (29308, 502)
stim_input_group_id = np.zeros((Fstim_interp.shape[1], num_unique_groups))  # EX: (29308, 100)
stim_continuous_input_id = np.zeros((Fstim_interp.shape[1], num_neurons))  # EX: (29308, 502)



# looping through 2046 non-zero stimulation t's 
for i in range(stim_steps.shape[0]):
    # stim_steps[i] is a t value (1,2,3,...29308)
    # setting the 3 t's after every stimulated t (t:t+3) to 1
    stim_input[stim_steps[i]: stim_steps[i] + interp_len] = 1

    # stimID_correct[t] is either 0 or the photostim group number
    # subtract each stim_group_id by 1 for indexing at later step, if get rid of - 1 there is a index out of bounds error
    stim_group_id = int(stimID_correct[stim_steps[i]] - 1) #74-1, 84-1, ... 26-1
    
    # only return neurons from each photostim group that are within 10 pixels of a beamlet, each iteration is a photostim group
    stim_neuron_ids = np.where(stimDist[:,stim_group_id]<stim_dist)[0]
    
    target_mask = np.zeros((num_neurons))
    target_mask[stim_neuron_ids]  = 1
    
    # setting the stimulated neurons from photostim group stim_group_id 
    # (which are listed in stim_neuron_ids, which is used to create the target_mask) 
    # to have the value of 1, or target_mask, for t:t+3 time steps
    stim_input_id[stim_steps[i]: stim_steps[i] + interp_len, :] = target_mask

    # exponential decay so that distances further from photostim beamlets 
    # will be smaller and distances closer to beamlets will be larger
    spatial_input = np.exp(-data_all['photostim']['stimDist'][:,stim_group_id]/10) * np.exp(-data_all['photostim']['slmDist'][:,stim_group_id]/200)

    # setting the stimulated neurons from photostim group stim_group_id
    # (which are in data_all['photostim']['stimDist'][:,stim_group_id]
    # to the spatial, exponential decayed info for t:t+3 time steps
    stim_continuous_input_id[stim_steps[i]: stim_steps[i] + interp_len, :] = spatial_input
    stim_input_group_id[stim_steps[i]: stim_steps[i] + interp_len, stim_group_id] = 1 


data_save = {
        # 1s for t:t+3, shape is (29308, 502)
        'u_session': stim_input_id,
        # continuous spatial for t:t+3  
        'u_spatial_session': stim_continuous_input_id,  
        # Fstim_interp is (502,29308) so .T is (29308, 502), this is the intensity of photostim
        'y_session': Fstim_interp_norm.T,  
        # centroidX: 502 (list of neurons) mean X pixel indices for each neuron
        'x1': np.asarray(data_all['photostim']['centroidX']),
        # centroidY: 502 (list of neurons) mean Y pixel indices for each neuron
        'x2': np.asarray(data_all['photostim']['centroidY']),
        # stimPosition: (10, 2, 100) (beamlet x 2 x photostim groups) XY coordinates of each photostim beamlet in each photostim group
        'o1': data[dataset_id]['stimPosition'][:,0,:],  # x coord of all beamlet across all photostim group
        'o2': data[dataset_id]['stimPosition'][:,1,:]}  # y coord of all beamlet across all photostim group

np.save('data/sample_photostim_'+ mouse + '_spatial_date_' + date + '.npy', data_save)



y_session = data_save['y_session']  #(29308, 502)
u_session = data_save['u_session']  #(29308, 502)

fig, axes = plt.subplots(2, 1, figsize=(12,6)) #fig stores the whole figure, axes stores the individual plots
time_wind_start = 840
time_wind_end = 1240
neuron_start = 0
neuron_end = 1000

axes[0].imshow(u_session[time_wind_start: time_wind_end, neuron_start:neuron_end].T)
axes[0].set_title('stimulation inputs')
axes[0].set_xlabel('time steps 840 to 1240')
axes[0].set_ylabel('neuron id')

im = axes[1].imshow(y_session[time_wind_start: time_wind_end, neuron_start:neuron_end].T)
fig.colorbar(im)
axes[1].set_title('in vivo calcium recording')
axes[1].set_xlabel('time steps 840 to 1240')
axes[1].set_ylabel('neuron id')
fig.savefig('photo_intro.pdf')


# nan represent removed interpolated steps in y
fig, axes = plt.subplots(2, 1, figsize=(20,10))
time_wind_start = 1000
time_wind_end = 2000
axes[0].plot(u_session[time_wind_start: time_wind_end].T)  #(502, 1000)
axes[0].set_title('stimulation inputs of all neurons during time 1000:2000')
axes[0].set_xlabel('neurons')
axes[0].set_ylabel('stimulated 1 or not 0')

axes[1].plot(y_session[time_wind_start: time_wind_end].T)  #(502, 1000)
axes[1].set_title('intensities of all neurons during time 1000:2000')
axes[1].set_xlabel('neurons')
axes[1].set_ylabel('intensity of photostim')
fig.savefig('photo_intro.pdf')
plt.show()