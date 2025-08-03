import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 80
import matplotlib.colors as colors
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy.linalg as la
import os
import seaborn as sns
import yaml
from sklearn.metrics import r2_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
from scipy.stats import norm
import statistics

os.chdir("C:/Users/andre/Desktop/active_neuron")
with open("aj_models/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

ark_order = config['ark_order']

### NAN INTERPOLATION in y_session ###


path = r"C:\Users\andre\Desktop\active_neuron\data\sample_photostim_59_spatial_date_070623.npy"
data = np.load(path, allow_pickle = True).item() #dict

for key in data.keys():
    print(key)

y_session = data['y_session']  #np.ndarray photostimulation intensity value (t, neurons)
u_session = data['u_session']  #np.ndarray t:t+3 is 1 for stimulated (t, neurons)
y_session_interp = y_session.copy()

for i in range(y_session.shape[1]):  #i loop over neurons
    nan_start = -1
    nan_stop = -1
    for j in range(y_session.shape[0]):  #j loop over t
        if nan_start == -1 and np.isnan(y_session_interp[j,i]):  #if no nan detected yet and 1 is detected
            nan_start = j - 1  # set nan_start to 1 t before detected index
        if nan_start != -1 and not np.isnan(y_session_interp[j,i]):  # if nan has been detected previously and no longer detected
            nan_stop = j  #set nan_stop to t
        if nan_start != -1 and nan_stop != -1: #if a valid nan interval has been detected
            slope = y_session_interp[nan_stop,i] - y_session_interp[nan_start,i]
            for k in range(nan_stop - nan_start - 1):  #linearly interpolate the detected nan interval, in the y_session data
                y_session_interp[nan_start + k + 1,i] = slope*k/(nan_stop-nan_start-1) + y_session_interp[nan_start,i]
            nan_start = -1
            nan_stop = -1

np.save('u_session.npy', u_session)
np.save('y_session_interp.npy', y_session_interp)




###PATTERN COUNTING, a pattern is the non-zero/stimulated neurons ###

patterns = []
pattern_count = []
pattern_idx = []
pattern_length = []

start = -10
for t in range(u_session.shape[0]):  #loop through t

    #if there is a single non_zero element in t-th row of u_session 
    #and if t is more than 4 units greater than start
    if np.sum(np.abs(u_session[t,:])) > 0 and t > start + ark_order:

        idx = np.linspace(0,u_session.shape[1]-1,u_session.shape[1]).astype(int)
        # print(f"{idx.shape} idx is a np.array of size n with the actual n's as the values")

        on = u_session[t,:] > 0
        # print(f"{on.shape} on is a np.array boolean mask of size n with trues and falses of if that neuron is above 0 or not (stimulated or not)")

        pattern = np.array(idx[on])
        # print(f"{pattern} pattern is now the neurons at t which were non-zero")

        start = t  #set start to the current trial we are on
        found = False
        for i in range(len(patterns)):

            # if pattern has been found already
            if len(pattern) == len(patterns[i]):
                if np.linalg.norm(pattern - patterns[i]) == 0:
                    pattern_count[i] += 1
                    pattern_idx[i].append(t)
                    found = True
                    break

        # if pattern has not been found yet
        if found is False:
            patterns.append(pattern)  #there are 100 patterns
            pattern_count.append(1)
            pattern_idx.append([t])  #append the trial as the pattern_idx
            pattern_length.append(len(pattern))

            # print(f"added this pattern: {pattern}")
            # print(f"pattern count is now {pattern_count}")
            # print(f"added pattern index {pattern_idx}")
            # print(f"pattern length is now {pattern_length}")
            # print(" ")

# print(f"patterns should be {len(patterns)} long")
# print(f"pattern_count is the number of times each pattern was found.{pattern_count}")
# print(f"pattern_idx is all the t's that each pattern was found at, so this is going to be 100 sub arrays of a bunch of t's {pattern_idx}")
# print(f"pattern_length is the lengths of all patterns found. {pattern_length}")

#counting how many patterns each neuron is involved in, store in neuron_pattern
neuron_pattern = np.zeros(u_session.shape[1])
for i in range(u_session.shape[1]):
    for p in patterns:
        if i in p:
            neuron_pattern[i] += 1

# plt.figure(figsize=(10, 6), dpi=80)
# plt.hist(pattern_count, bins=[0,5,10,15,20,25,30,35])
# plt.title("Pattern Count vs Count Occurrence")
# plt.xlabel('Pattern Count: number of t a pattern was found')
# plt.ylabel('Count Occurrence: number of times the count itself occured')
# plt.show()

#(20, 10) means there are 10 unique patterns 
#that occurred exactly 20 times across all trials






idx = np.linspace(0,u_session.shape[1]-1,u_session.shape[1]).astype(int)
# print(f"{idx.shape} idx is a np.array of size n with the actual n's as the values")

# This will be 502 trues and falses
on = u_session[1,:] > 0
# print(f"{on.shape} on is a np.array of size n with trues and falses of if that neuron is above 0 or not")
# print(idx[on])
# print(f"{pattern} pattern is now the neurons at t which were non-zero")





### TRAIN-TEST SPLIT ###

#- remove 5 patterns each which looks like [63, 65, 95, 98, 124, 265, 279, 312, 367, 386, 399, 403, 484, 486, 489].

#- loop through every t that each pattern happens at which looks like [1, 1331, 3293, 3423, 6683, 7027, 10538, 14364, 16006, 19633, 20631, 21900, 23637, 24467, 24966, 25853, 26282, 26354, 27601]
#  and remove 20 t steps before it and 50 t steps after it



# remove patterns randomly
# ??? Why only 5 patterns?
num_patterns = 5
removed_patterns = []
removed_neurons = []
removed_steps = np.zeros(u_session.shape[0])
removed_count = 0

np.random.seed(0)
while len(removed_patterns) < num_patterns:  # while the patterns are NOT all removed
    p_idx = np.random.randint(0,len(patterns))  #pick random pattern index, 1-100
    # print('pattern id:', p_idx)
    if p_idx not in removed_patterns:
        removed_patterns.append(p_idx)
        removed_count += pattern_count[p_idx]  # += will be a number like 22
        # print(f"removed pattern indexes are now {removed_patterns}")
        # print(f"total removed pattern count is now {removed_count}")
        
        #add each [1,99,500, etc.] pattern of neurons at p_idx to removed_neurons
        removed_neurons.extend(patterns[p_idx])
        # print(f"total removed neurons are now {removed_neurons}")

        #looping through the t's the pattern at p_idx occurred at
        #remove interval from 20 before t to 50 after t
        for i in pattern_idx[p_idx]:  
            min_idx = np.max([0,i-20])
            max_idx = np.min([i+50,u_session.shape[0]-1])
            #marked all t's in this interval for removal with a 1
            removed_steps[min_idx:max_idx] = 1

print(f"print the number of removed t's {np.sum(removed_steps)}")
print(f"print total removed pattern count {removed_count} and print the fraction of removed t's {np.sum(removed_steps) / u_session.shape[0]}")
print(f"print the removed pattern indexes {removed_patterns}")
plt.figure(figsize=(20,5))
plt.plot(removed_steps)
plt.title('Train-Test Split')

#set the removed steps as the test set
test_indices = removed_steps.astype(int)

#set the non removed steps as the train set
train_indices = np.ones(test_indices.shape[0]).astype(int) - test_indices.copy()

removed_neurons = list(set(removed_neurons))
np.save('removed_neurons', removed_neurons)

# print(f"EX of pattern_idx[p_idx] will be a series of t's where this pattern occurred {pattern_idx[44]}")
# print(f"{len(pattern_idx[44] + pattern_idx[44] + pattern_idx[47] + pattern_idx[64] + pattern_idx[67] + pattern_idx[9])} t's and intervals of t-20:t+50 for each one of these was marked for test set ~roughly, some at end left out")

print(f"train set is {np.sum(train_indices)} 1's in a np.array {len(train_indices)} long.")
print(f"test set is {np.sum(test_indices)} 1's in a np.array {len(test_indices)} long.")

train_test_split = {}
train_test_split['train_indices'] = train_indices
train_test_split['test_indices'] = test_indices

np.save('train_test_split',train_test_split)






### PLOT THE TRUE SPIKES ###

spiking = np.zeros(u_session.shape).astype(bool)
# print(f"{spiking.shape} spiking is boolean matrix of shape u_session.shape")

for neuron in range(u_session.shape[1]):
    output_true = y_session_interp[ark_order:,neuron]
    #baseline intensity level for neuron
    mean_threshold = np.median(output_true)

    #idxs of t's where activity is lower than baseline
    lower_tail_idx = (output_true < mean_threshold)

    lower_tail_data = output_true[lower_tail_idx]

    lower_tail_std = np.std(lower_tail_data)

    true_spike_threshold = mean_threshold + 6*lower_tail_std #identify spikes
    true_spikes = (output_true > true_spike_threshold)
    spiking[ark_order:,neuron] = true_spikes  

# print(f"{mean_threshold} is the median intensity value for neuron {neuron}")
# print(f"{lower_tail_std} is the lower tail std dev for neuron {neuron}'s intensity values")
# print(f"{np.sum(lower_tail_idx)} / {len(lower_tail_idx)} is the proportion of lower tail for neuron {neuron}")
# print(f"{lower_tail_data} these are the actual intensity values of the lower tail for neuron {neuron}")

# print(f"{true_spikes} true_spikes is a boolean array of if an intensity reading at t is considered a spike or not")
# print(f"{np.sum(true_spikes)} / {len(true_spikes)} are considered true spikes for neuron {neuron}")
#np.save('spiking_labels',spiking)




neuron = 206
T = 1000
plt.figure(figsize=(12,6))
plt.plot(y_session_interp[:T,neuron])
idx = np.linspace(0,T-1,T)
spiking_neuron = spiking[:T,neuron]  # (1000, 1) boolean spike or not for 1st 1000 t
y_out = y_session_interp[:T,neuron].copy()  # (1000, 1) actual intensity values for 1st 1000 t

y_out = y_out * spiking_neuron  # 0 out non-spikes
y_out[y_out == 0] = np.nan  # nan out the non-spikes
plt.scatter(idx,y_out,color='r')  
plt.show()