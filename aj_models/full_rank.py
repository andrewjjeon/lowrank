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
import pickle
from sklearn.metrics import r2_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
from low_rank_model import (
    to_np, filter_indices_with_lags, split_dataset, TimeseriesDataset, 
    singular_value_norm, LinearDynamicModel, LowRankLinearDynamicModel, 
    train_model, low_rank_svd_components_approximation, diag_off_diag_extraction, 
    compute_transfer_matrix, plot_rank_svd
)

################################### full_rank.py CONTENTS ######################################
# - nan interpolation
# - train-test split
# - true spikes
# - closed-form solution Ahat
# - closed-form Ahat TPR/FPR
# - closed-form Ahat Test MSE
# - clsoed-form Causal connectivity matrix
#

############################# SAMPLE DATA DESCRIPTION ######################################
# ------------------------------------------------------------------------------------------
# u_session:         EX: (29308,502) = (t, neurons) 1's for t:t+3, 1 for stimulation event, 
# u_spatial_session: EX: (29308,502) continuous distance from beamlet 
# y_session:         EX: (29308,502) intensity of photostim
# x1:                mean X pixel indices for each neuron
# x2:                mean Y pixel indices for each neuron
# o1:                x coord of all beamlet across all photostim group
# o2:                y coord of all beamlet across all photostim group
os.chdir("C:/Users/andre/Desktop/active_neuron")
with open("aj_models/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

normalize = config['normalize']
ark_order = config['ark_order']

mouse = config['mouse']
date = config['date']

try:
    data = np.load('data/sample_photostim_'+ mouse + '_spatial_date_' + date + '.npy', allow_pickle = True).item()
    print("Keys of raw data: ", data.keys())
except FileNotFoundError:
    print("This is not a valid mouse + date combination. Please try again.")

# NAN INTERPOLATION
y_session = data['y_session']  # photostim intensity value (t, neurons)
u_session = data['u_session']  # t:t+3 is 1 for stimulated (t, neurons)
y_session_interp = y_session.copy()


num_neurons = u_session.shape[1]

for i in range(y_session.shape[1]):  # loop over neurons
    nan_start = -1
    nan_stop = -1
    for j in range(y_session.shape[0]):  # loop over t

        # if no nan detected yet and 1 is detected
        if nan_start == -1 and np.isnan(y_session_interp[j,i]):
            nan_start = j - 1  # set nan_start to 1 t before detected index

        # if nan has been detected previously and no longer detected
        if nan_start != -1 and not np.isnan(y_session_interp[j,i]):
            nan_stop = j  #set nan_stop to t

        #if a valid nan interval has been detected
        if nan_start != -1 and nan_stop != -1:
            slope = y_session_interp[nan_stop,i] - y_session_interp[nan_start,i]

            #linearly interpolate the detected nan interval, in the y_session data
            for k in range(nan_stop - nan_start - 1):
                y_session_interp[nan_start + k + 1,i] = slope*k/(nan_stop-nan_start-1) + y_session_interp[nan_start,i]
            nan_start = -1
            nan_stop = -1
            
np.save('u_session.npy', u_session)
np.save('y_session_interp.npy', y_session_interp)

# PATTERN COUNTING
# a pattern is the non-zero/stimulated neurons

patterns = []
pattern_count = []
pattern_idx = []
pattern_length = []

start = -10
for t in range(u_session.shape[0]):  #loop through t

    # if there is a single non_zero element in t row (502,) of u_session 
    # and if t is more than 4 time steps past start
    if np.sum(np.abs(u_session[t,:])) > 0 and t > start + 4:

        # idx is a np.array of size n with the actual n's as the values
        idx = np.linspace(0,u_session.shape[1]-1,u_session.shape[1]).astype(int)

        # on is a np.array boolean mask of size n with trues and falses 
        # of if that neuron is above 0 or not (stimulated or not)
        on = u_session[t,:] > 0

        # pattern is now the neurons at t which were non-zero
        pattern = np.array(idx[on])

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
            # add pattern ~ roughly 100 patterns at end
            patterns.append(pattern)

            # pattern_count is the number of times each pattern was found
            pattern_count.append(1)

            # pattern_idx is all the t's that each pattern was found at, 
            # so this is going to be 100 sub arrays of a bunch of t's
            pattern_idx.append([t])

            # pattern_length is the lengths of all patterns found
            pattern_length.append(len(pattern))

#counting how many patterns each neuron is involved in, store in neuron_pattern
neuron_pattern = np.zeros(u_session.shape[1])
for i in range(u_session.shape[1]):
    for p in patterns:
        if i in p:
            neuron_pattern[i] += 1

############################ TRAIN-TEST SPLIT PROCESSING ###################################
# nan interpolation
# TRAIN-TEST SPLIT VIA PATTERN REMOVAL
#   - remove 5 patterns each which looks like 
#     [63, 65, 95, 98, 124, 265, 279, 312, 367, 386, 399, 403, 484, 486, 489].
#
#   - loop through every t that each pattern happens at which looks like 
#     [1, 1331, 3293, 3423, 6683, 7027, 10538, 14364, 16006, 19633, 20631, 21900, 
#      23637, 24467, 24966, 25853, 26282, 26354, 27601]
#     and remove 20 t steps before it and 50 t steps after it
#
#


num_patterns = 5
removed_patterns = []
removed_neurons = []

removed_steps = np.zeros(u_session.shape[0])
removed_count = 0

np.random.seed(0)
while len(removed_patterns) < num_patterns:  # while the patterns are NOT all removed
    p_idx = np.random.randint(0,len(patterns))  # pick random pattern index, 1-100
    if p_idx not in removed_patterns:
        removed_patterns.append(p_idx)  # append the removed pattern index

        # append the # of times the pattern at p_idx was found
        removed_count += pattern_count[p_idx]
        
        # add each removed pattern of neurons at p_idx to removed_neurons
        # [63, 65, 95, 98, 124, 265, 279, 312, 367, 386, 399, 403, 484, 486, 489]
        removed_neurons.extend(patterns[p_idx])

        #looping through the t's the pattern at p_idx occurred at
        #remove interval from 20 before t to 50 after t
        for i in pattern_idx[p_idx]:  
            min_idx = np.max([0,i-20])
            max_idx = np.min([i+50,u_session.shape[0]-1])
            #marked all t's in this interval for removal with a 1
            removed_steps[min_idx:max_idx] = 1

plt.figure(figsize=(15,6), dpi=80)
plt.plot(removed_steps)
plt.title("Train-Test Split Visualized")

#set the removed steps as the test set
test_indices = removed_steps.astype(int)
np.save('test_indices', test_indices)

#set the non removed steps as the train set
train_indices = np.ones(test_indices.shape[0]).astype(int) - test_indices.copy()
np.save('train_indices.npy', train_indices)
removed_neurons = list(set(removed_neurons))

with open('removed_neurons.pkl', 'wb') as f:
    pickle.dump(removed_neurons, f)

print(f"train set is {np.sum(train_indices)} 1's in a np.array {len(train_indices)} long.")
print(f"test set is {np.sum(test_indices)} 1's in a np.array {len(test_indices)} long.")

train_test_split = {}
train_test_split['train_indices'] = train_indices
train_test_split['test_indices'] = test_indices
# np.save('train_test_split',train_test_split)



########################## ACCUMULATE TRUE SPIKES FOR EACH NEURON ##############################

spiking = np.zeros(u_session.shape).astype(bool)

for neuron in range(u_session.shape[1]):
    output_true = y_session_interp[4:,neuron]
    # threshold is median intensity level for each neuron
    mean_threshold = np.median(output_true)

    #idxs of t's where activity is lower than baseline
    lower_tail_idx = (output_true < mean_threshold)
    lower_tail_data = output_true[lower_tail_idx]
    lower_tail_std = np.std(lower_tail_data)
    true_spike_threshold = mean_threshold + 6*lower_tail_std #identify spikes
    true_spikes = (output_true > true_spike_threshold)
    spiking[4:,neuron] = true_spikes

# PLOT A NEURONS TRUE SPIKES
neuron = 206
T = 1000
plt.figure(figsize=(12,6), dpi=80)
plt.plot(y_session_interp[:T,neuron])
idx = np.linspace(0,T-1,T)
spiking_neuron = spiking[:T,neuron]  # (1000, 1) boolean spike or not for 1st 1000 t
y_out = y_session_interp[:T,neuron].copy()  # (1000, 1) actual intensity values for 1st 1000 t

y_out = y_out * spiking_neuron  # 0 out non-spikes
y_out[y_out == 0] = np.nan  # nan out the non-spikes

plt.title("A Neuron's true spikes")
plt.xlabel("time")
plt.ylabel("photostim intensity")
plt.scatter(idx,y_out,color='r')  


############################ CLOSED-FORM LINEAR REGRESSION ###########################

X = []
Xp = []
# train_indices here is 29308, full length
for t in range(train_indices.shape[0]-ark_order):  # 29304
    # if it is valid training data only 22636 valid training indices
    if np.sum(train_indices[t:t+ark_order+1]) == ark_order + 1 and t > 5 + ark_order:
        # X.append( (4 * 502) + (4 * 502) + 1 = 4017) 
        # X is all the (concatted previous t:t+4 x's and t:t+4 u's) appended for each t
        # Xp.append( (502) ) = Xp is all the t+4 states appended; the t+4 state is the state we want to predict
        X.append(np.concatenate((y_session_interp[t:t+ark_order,:].copy().flatten()/normalize, u_session[t:t+ark_order,:].copy().flatten(),np.ones(1))))  # concatenating a 1 for y = Xw + b, bias term
        Xp.append(y_session_interp[t+ark_order,:].copy().flatten()/normalize)

X = np.array(X)  # (22636, 4017)
Xp = np.array(Xp)  # (22636, 502) basically our y

# W = (X.T @ X)^-1 @ X.T @ Y
# np.linalg.pinv calculates the inverse of a matrix
# Ahat is basically the weight matrix used to make predictions
Ahat = np.linalg.pinv(X.T @ X) @ X.T @ Xp  # (4017,502)
Ahat = Ahat.T  # (502,4017)
np.save('Ahat.npy', Ahat)


######################### CLOSED-FORM LINEAR REGRESSION TEST SET MSE ############################
x_pred = []
x_true = []
u_true = []
r2 = [] 
idx = -1

new_segment = True
segment_pred = []
segment_start = -1
for t in range(train_indices.shape[0]):  #29308 t's
    if test_indices[t] == 1:  # if it's test set
        if new_segment:  # if it's new segment
            segment_pred = []
            new_segment = False
            segment_start = t
            x_past = []
            x_pred.append([])
            x_true.append([])
            u_true.append([])
            idx += 1  # marks each segment, 82 segments found
        if t < segment_start + ark_order:  # if current time step is within first 'ark_order" (k in paper) t's of segment start
            x_past.append(y_session_interp[t,:].copy().flatten()/normalize)  # append (502) neuron state at t to x_past
        
        #if t is in test set, if not new_segment, and if t is outside k of segment start
        else:

            z = np.array(x_past).flatten()  # z should be a full ark order y_session[t:t+4,:] flattened (2008, )
            #concat z with the corresponding (t:t+4) u_session data flattened and a 1 for the bias term, (4017, )
            z = np.concatenate((z,u_session[t-ark_order:t,:].copy().flatten(),np.ones(1)))

            # (502, 4017) @ (4017, ) = (502, ) x_next
            x_next = Ahat @ z
            x_past.pop(0) #pop oldest (502,)
            x_past.append(x_next.copy()) # append x_next to x_past to continue making predictions for next t
            x_pred[idx].append(x_next.copy()) # append predicted value to that segment x_pred
            x_true[idx].append(y_session_interp[t,:].copy().flatten()/normalize)
            u_true[idx].append(u_session[t,:].copy().flatten())
    else: #if current t is not part of test set, a new segment should start
        new_segment = True

mse_losses = []
total_t_in_segs = 0
for i in range(len(x_pred)):  # for each segment in segments 82 segments, likely all different lengths
    # print(len(x_pred[i]))
    total_t_in_segs += len(x_pred[i])
    x_pred[i] = np.array(x_pred[i])  # (66, 502), (109, 502), ... (segment length, 502) etc.
    x_true[i] = np.array(x_true[i])
    u_true[i] = np.array(u_true[i])  
    mse_losses.append((np.square(x_pred[i] - x_true[i])).mean())  # x_pred[i] is 66 predicted t's - x_true[i] 66 true t's
    r2.append(r2_score(x_true[i],x_pred[i]))

print('Closed-Form Linear Regression MSE:', sum(mse_losses)/len(mse_losses))
print('Closed-Form Linear Regression R2:', sum(r2)/len(r2))



####################### TPR/FPR for Closed-Form Linear Regression model #######################
tpr = []
fpr = []
thresholds = np.linspace(-2,5,15)

for thresholds_idx in range(len(thresholds)):  # 15 thresholds
    tp_total = 0
    fp_total = 0
    p_total = 0
    n_total = 0

    for neuron in range(u_session.shape[1]):  # 502 neurons
        output_pred = []
        output_true = []
        for i in range(len(x_true)):  # 82 segments
            output_pred.extend(x_pred[i][:,neuron])  # (66,1), (109,1), ... (len(segment), 1)
            output_true.extend(x_true[i][:,neuron])

        output_pred = np.array(output_pred)  # (6012, 1), 6012 t's, output_pred and output_true for each neuron now
        output_true = np.array(output_true)  # (6021, 1)
    
        mean_threshold = np.median(output_true)  # set mean threshold to be the median true intensity of current neuron 
        lower_tail_idx = (output_true < mean_threshold)  # below threshold indices/t's
        lower_tail_data = output_true[lower_tail_idx]  # intensities at below threshold indices
        lower_tail_std = np.std(lower_tail_data)
        true_spike_threshold = mean_threshold + 6*lower_tail_std
        detect_spike_threshold = mean_threshold + thresholds[thresholds_idx]*lower_tail_std

        predicted_spikes = (output_pred > detect_spike_threshold)  # how many t's out of 6012 are predicted spike
        true_spikes = (output_true > true_spike_threshold)  # how many t's out of 6012 are true spike
        tp_total += np.sum(np.logical_and(predicted_spikes,true_spikes))
        fp_total += np.sum(np.logical_and(predicted_spikes,~true_spikes))
        p_total += np.sum(true_spikes)
        n_total += np.sum(~true_spikes)

    # At this point tp_total, fp_total, p_total, n_total contain values from all 502 neurons and all their spikes across 82 segments worth of t's
    # We add the tpr and fpr per threshold here
    tpr.append(tp_total / p_total)
    fpr.append(fp_total / n_total)

tpr_noninput = []
fpr_noninput = []
thresholds = np.linspace(-2,5,15)

for thresholds_idx in range(len(thresholds)):  # 15 thresholds
    tp_total = 0
    fp_total = 0
    p_total = 0
    n_total = 0

    for neuron in range(u_session.shape[1]):  # 502 neurons 
        output_pred = []
        output_true = []
        for i in range(len(x_true)): # 82 segments
            #KEY DIFF: if this neuron was unexcited, then do all same --> this excludes the excited neurons
            if np.sum(u_true[i][:,neuron]) == 0:
                output_pred.extend(x_pred[i][:,neuron])   # (66,1), (109,1), ... (len(segment), 1)
                output_true.extend(x_true[i][:,neuron])

        output_pred = np.array(output_pred)  # (6012, 1), 6012 t's, output_pred and output_true for each neuron now
        output_true = np.array(output_true)
    
        mean_threshold = np.median(output_true)  # set mean threshold to be the median true intensity of current neuron 
        lower_tail_idx = (output_true < mean_threshold)  # below threshold indices/t's
        lower_tail_data = output_true[lower_tail_idx]  # intensities at below threshold indices
        lower_tail_std = np.std(lower_tail_data)
        true_spike_threshold = mean_threshold + 6*lower_tail_std
        detect_spike_threshold = mean_threshold + thresholds[thresholds_idx]*lower_tail_std

        predicted_spikes = (output_pred > detect_spike_threshold)
        true_spikes = (output_true > true_spike_threshold)
        tp_total += np.sum(np.logical_and(predicted_spikes,true_spikes))
        fp_total += np.sum(np.logical_and(predicted_spikes,~true_spikes))
        p_total += np.sum(true_spikes)
        n_total += np.sum(~true_spikes)

    tpr_noninput.append(tp_total / p_total)
    fpr_noninput.append(fp_total / n_total)

results = {}
results['fpr'] = fpr
results['tpr'] = tpr
results['fpr_noninput'] = fpr_noninput
results['tpr_noninput'] = tpr_noninput
np.save('results/results_full.npy', results)


results_load = np.load('results/results_full.npy', allow_pickle=True).item()

# full data linear model vs non-input neurons only
plt.figure(figsize=(8,6), dpi=80)
plt.plot(results_load['fpr_noninput'],results_load['tpr_noninput'],label='unexcited-linear')
plt.plot(results_load['fpr'],results_load['tpr'],label='all-linear')
plt.title('TPR and FPR for detected spikes for Closed-Form Linear Regression')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()


##################### Closed-Form Linear Regression Performance PLOT ######################
neuron = removed_neurons[9]
length = 50
t_start = 30

win_len = 25  # observe only 30:55 segments

output_pred = []
output_true = []
input_true = []
segment_marker = []

for i in range(t_start, t_start + win_len):  # looping through 25 segments
    # add all segment's in the interval's neuron data 
    output_pred.extend(x_pred[i][:,neuron])  # segment i's predicted t's data for neuron added to new output_pred
    output_true.extend(x_true[i][:,neuron])  # segment i's true t's data for neuron
    input_true.extend(u_true[i][:,neuron])

    segment_marker.extend(np.nan*np.zeros(len(x_pred[i][:,neuron])-1))  # everything before last t in segment as nan
    segment_marker.extend([0.25,0])  # add 0.25 followed by 0 to mark end of segment
    

# This plot is basically 25 segments (uninterrupted series of t's within k interval from t start) 
# (each segment is 66 at least and can be larger) and their true and predicted  intensity values stitched together
plt.figure(figsize=(12,6), dpi=80)
plt.subplot(2,1,1)
plt.plot(output_true,label='truth')
plt.plot(output_pred,label='predicted')
#plt.plot(segment_marker)
plt.title('Closed-Form Linear Regression Model Performance for a neuron')
plt.legend()

plt.subplot(2,1,2)
plt.plot(input_true,label='input')
plt.title('Input')



################## COMPUTE CAUSAL CONNECTIVITY MATRIX FOR CLOSED-FORM AHAT #####################

# A = compute_transfer_matrix(Ahat, num_neurons) # causal connectivity matrix
# plt.figure(figsize=(20,15), dpi=80)
# plt.subplot(2,2,1)
# plt.title('causal_connectivity_matrix')
# plt.imshow(A)
# plt.clim([-0.2, 0.2])

# plt.subplot(2,2,2)
# plt.title('causal_connectivity_matrix_minus_diagonal')
# plt.imshow(A - np.diag(np.diag(A))) # connectivity matrix - its diagonal

# # mean of the largest singular values is used to determine connection coefficient
# connection_threshold = 1*np.mean(np.diag(A)) 
# A_threshold = (A > connection_threshold).astype(float)
# plt.subplot(2,2,3)
# plt.title('Causal_connectivity_matrix_above_threshold')
# plt.imshow(A > connection_threshold)

# plt.figure(figsize=(10,5), dpi=80)
# plot_rank_svd(A)

plt.show()