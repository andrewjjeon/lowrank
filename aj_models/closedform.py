import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.dpi'] = 200
import matplotlib.colors as colors
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy.linalg as la
import os
import yaml
import pickle
import seaborn as sns
from sklearn.metrics import r2_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
from scipy.stats import norm
import statistics
from low_rank_model import (
    to_np, filter_indices_with_lags, split_dataset, TimeseriesDataset, 
    singular_value_norm, LinearDynamicModel, LowRankLinearDynamicModel, 
    train_model, low_rank_svd_components_approximation, diag_off_diag_extraction, 
    compute_transfer_matrix, plot_rank_svd
)




os.chdir("C:/Users/andre/Desktop/active_neuron")
with open("aj_models/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

u_session = np.load('u_session.npy')
y_session_interp = np.load('y_session_interp.npy')

train_test_split = np.load('train_test_split.npy', allow_pickle=True).item()
train_indices = train_test_split['train_indices']
test_indices = train_test_split['test_indices']
# print(train_test_split)
# print(len(train_indices))
# print(len(test_indices))

ark_order = config['ark_order']
normalize = config['normalize']
# print(ark_order)
# print(normalize)

removed_neurons = np.load('removed_neurons.npy')





### CLOSED-FORM LINEAR REGRESSION ###

print(f"there are {len(train_indices)} t's total")
print(f"{np.sum(train_indices)} of those are training t's")
print(f"{np.sum(test_indices)} of those are test t's")

X = []
Xp = []
# train_indices here is 29308, full length
for t in range(train_indices.shape[0]-ark_order):  # 29304
    if np.sum(train_indices[t:t+ark_order+1]) == ark_order + 1 and t > 5 + ark_order:  #if it is valid training data only 22636 valid training indices
        # X.append( (4 * 502) + (4 * 502) + 1 = 4017) 
        # X is all the (concatted previous t:t+4 x's and t:t+4 u's) appended for each t
        # Xp.append( (502) ) = Xp is all the t+4 states appended; the t+4 state is the state we want to predict
        X.append(np.concatenate((y_session_interp[t:t+ark_order,:].copy().flatten()/normalize, u_session[t:t+ark_order,:].copy().flatten(),np.ones(1))))  # concatenating a 1 for y = Xw + b, bias term
        Xp.append(y_session_interp[t+ark_order,:].copy().flatten()/normalize)

print(f"There are only {len(X)} valid training indices or t's.")
X = np.array(X)  # (22636, 4017)
Xp = np.array(Xp)  # (22636, 502) basically our y
print(f"for t in 22636 valid training t's: X.append( (4 * 502) + (4 * 502) + 1 ) to get X with shape {X.shape}")
print(f"for t in 22636 valid training t's Xp.append( (502) ) to get X_pred with shape{Xp.shape}")

# W = (X.T @ X)^-1 @ X.T @ Y
# np.linalg.pinv calculates the inverse of a matrix
# Ahat is basically the weight matrix used to make predictions
Ahat = np.linalg.pinv(X.T @ X) @ X.T @ Xp
print(f"{Ahat.shape} Ahat relates the 4017 full segment of y_session + u_session + b to y_pred ")
Ahat = Ahat.T
print(f"{Ahat.shape} Ahat.T")  # transposing to be able to multiply below
np.save('Ahat.npy', Ahat)






# Test set evaluation (MSE) for Closed-Form Linear Regression Model

# this block uses a segment of 4 connected t's of y_session + u_session + b to predict x_next t
x_pred = []
x_true = []
u_true = []
r2 = [] 
idx = -1
new_segment = True
segment_pred = []
segment_start = -1
for t in range(train_indices.shape[0]):  #29308 t's
    if test_indices[t] == 1:  #if it's test set, test set has 6340 t's
        if new_segment:  # if it's new segment
            segment_pred = []
            new_segment = False
            segment_start = t
            x_past = []
            x_pred.append([])
            x_true.append([])
            u_true.append([])
            idx += 1  # marks each segment, 82 segments found
        if t < segment_start + ark_order:  #if current time step is within first 'ark_order" (k in paper) t's of segment start
            x_past.append(y_session_interp[t,:].copy().flatten()/normalize)  # append (502) neuron state at t to x_past
        
        #if t is in test set, if not new_segment, and if t is outside k of segment start
        else:
            # print(f"x_past is this shape when t is outside k of segment start{len(x_past)}")  # 4
            # print(f" this is the current segment {idx}")  # 0
            z = np.array(x_past).flatten()  # z should be a full ark order y_session[t:t+4,:] flattened (2008, )
            # print(f"z should be a full segment of y_session[t:t+4,:] flattened {z.shape}")

            #concat z with the corresponding (t:t+4) u_session data flattened and a 1 for the bias term, (4017, )
            z = np.concatenate((z,u_session[t-ark_order:t,:].copy().flatten(),np.ones(1)))
            # print(f"z should be a full segment of y_session[t:t+4,:] flattened concatted with u_session[t:t+4,:] flattened concatted with a 1 {z.shape}")

            # (502, 4017) @ (4017, ) = (502, ) x_next
            x_next = Ahat @ z
            x_past.pop(0) #pop oldest (502,)
            x_past.append(x_next.copy()) # append x_next to x_past to continue making predictions for next t
            x_pred[idx].append(x_next.copy()) # append predicted value to that segment x_pred
            x_true[idx].append(y_session_interp[t,:].copy().flatten()/normalize)
            u_true[idx].append(u_session[t,:].copy().flatten())
    else: #if current t is not part of test set, a new segment should start
        new_segment = True

print(f"x_pred shape {len(x_pred)}")
print(f"x_true shape {len(x_true)}")
print(f"u_true shape {len(u_true)}")



# print(len(x_pred[0]))
# print(len(x_pred[40]))
# print(f"""there are 82 segments many of which have a length of {len(x_pred[81])}, some segments will be longer, but none will be shorter. 66 makes sense, 
# because when splitting test set earlier, 20 t's before and 50 t's after --> 70 t's a piece were marked for removal to test set, however we only add to 
# a segment when a t is more than k=4 t's outside segment start. Segments are unbroken series of t's outside ark order of whawtever index we are starting at.""")

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

print(f"{total_t_in_segs} t's in all segments combined")
print('mse:', sum(mse_losses)/len(mse_losses))
print('r2:', sum(r2)/len(r2))







# TPR/FPR for Closed-Form Linear Regression model

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
  
        # TODO: we seem to be calculating TPR and FPR somewhat arbitrarily, mainly i don't know where 6 for true_spike_threshold and trying out np.linspace(-2,5,15) thresholds for the detect_spike_threshold comes from
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

# print(f"{tpr} are the CFS Linear models TPR's for the different threshold values tried")
# print(f"{fpr} are the CFS Linear models FPR's for the different threshold values tried")

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

    # At this point tp_total, fp_total, p_total, n_total contain values from all 502 neurons and all their spikes across 82 segments worth of t's
    # We add the tpr and fpr per threshold here
    tpr_noninput.append(tp_total / p_total)
    fpr_noninput.append(fp_total / n_total)

results = {}
results['fpr'] = fpr
results['tpr'] = tpr
results['fpr_noninput'] = fpr_noninput
results['tpr_noninput'] = tpr_noninput
np.save('results/results_full.npy', results)



results_load = np.load('results/results_full.npy', allow_pickle=True).item()

sorted_indices = np.argsort(results_load['fpr'])
fpr_sorted = np.array(results_load['fpr'])[sorted_indices]
tpr_sorted = np.array(results_load['tpr'])[sorted_indices]
sorted_indices_noninput = np.argsort(results_load['fpr_noninput'])
fpr_noninput_sorted = np.array(results_load['fpr_noninput'])[sorted_indices_noninput]
tpr_noninput_sorted = np.array(results_load['tpr_noninput'])[sorted_indices_noninput]

# Calculate AUROC for both curves using numerical integration
auroc = np.trapz(tpr_sorted, fpr_sorted)
auroc_noninput = np.trapz(tpr_noninput_sorted, fpr_noninput_sorted)

# # Store the AUROC values
# aurocs.append(auroc)
# aurocs_noninput.append(auroc_noninput)

print(f"AUROC is {auroc}")
print(f"AUROC_noninput is {auroc_noninput}")



results_load = np.load('results/results_full.npy', allow_pickle=True).item()

# full data linear model vs non-input neurons only
plt.figure(figsize=(8,6), dpi=80)
plt.plot(results_load['fpr_noninput'],results_load['tpr_noninput'],label='unexcited-linear')
plt.plot(results_load['fpr'],results_load['tpr'],label='all-linear')
plt.title('TPR and FPR for spikes detected')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()





# Closed-Form Linear Regression Model
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
plt.figure(figsize=(25,15), dpi=80)
plt.subplot(2,1,1)
plt.plot(output_true,label='truth')
plt.plot(output_pred,label='predicted')
#plt.plot(segment_marker)
plt.title('Closed-Form Linear Regression Model')
plt.legend()

plt.subplot(2,1,2)
plt.plot(input_true,label='input')
plt.title('Input')
plt.savefig('results/CFS_Linear_Model.pdf')