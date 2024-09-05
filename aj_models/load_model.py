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


os.chdir("C:/Users/andre/Desktop/active_neuron")

# Load the config file
with open("aj_models/config.yaml", 'r') as file:
    config = yaml.safe_load(file)


rank_dim = config['rank_dim']
state_dim = config['state_dim']
input_dim = config['input_dim']

ark_order = config['ark_order']
num_lags = config['ark_order']
init_toggle = config['init_toggle']

num_neurons = config['num_neurons']
batch_size = config['batch_size']  # how many samples/t's/indices per batch to load in DataLoader
normalize = config['normalize']
epochs = config['epochs']
learning_rate = config['learning_rate']
lambda_val = config['lambda_val']

u_session = np.load('u_session.npy')
y_session_interp = np.load('y_session_interp.npy')

train_test_split = np.load('train_test_split.npy', allow_pickle=True).item()
train_indices = train_test_split['train_indices']
test_indices = train_test_split['test_indices']


removed_neurons = np.load('removed_neurons.npy')


# comment out the init_value blocks if you want to initialize with nothing, instead of Ahat matrix

with open('init_value.pkl', 'rb') as file:
    init_value = pickle.load(file)

# init_value = None

#################################### LOAD THE BEST MODEL CHECKPOINT ##############################
# Initialize the model
model = LowRankLinearDynamicModel(state_dim, input_dim, rank_dim, num_lags, init_value)
checkpoint_name = 'low_rank_' + str(rank_dim) + '_epochs_' + str(epochs) + '_lr_' + str(learning_rate) + '_lambda_' + str(lambda_val) + '_k_' + str(ark_order) + '_init_value_' + init_toggle
checkpoint_path = 'checkpoints/model_best_' + checkpoint_name + '.pt'  # Update the path and filename as needed
model.load_state_dict(torch.load(checkpoint_path))  # load the model that had the lowest val loss
model.eval()  # Set the model to evaluation mode

init_value = {}
init_value['alpha'] = []
init_value['W_u'] = []
init_value['W_v'] = []
init_value['beta'] = []
init_value['B_u'] = []
init_value['B_v'] = []
init_value['V'] = []

Ahat_list = [] #will concatenate this list, W/A, B, and V

# expand and convert torch.Size([502]) to np (502, 1), and insert it at the top
# by the end, v will be on bottom, above it will be B, above it will be A
Ahat_list.insert(0, np.expand_dims(to_np(model.V), axis = 1))
init_value['V'].insert(0, to_np(model.V))


#(502=neuron state of x_next, 2008:2510, 2510:3012, 3012:3514, 3514:4016 previous inputs u_session)
for Bu_k, Bv_k, betak in zip(model.B_u, model.B_v, model.beta):
    init_value['beta'].insert(0, to_np(betak))  # OG diagonals of each (502,502) CFS Linear Ahatmatrix B slice, before low rank
    init_value['B_u'].insert(0, to_np(Bu_k))  # U vectors for each low rank B matrix of each 
    init_value['B_v'].insert(0, to_np(Bv_k))  # V vectors time S singular values for each low rank B matrix
    Bk = torch.mm(Bu_k, Bv_k.T)  # reconstruct each low rank B matrix, (502,5) @ (5,502) = (502,502)

    # add the corresponding pre-low rank CFS Linear Ahat OG diagonals back to each reconstructed low_rank matrices
    # insert the B = U @ V.T + D (502,502) to top of Ahat
    Ahat_list.insert(0, to_np(singular_value_norm(Bk + torch.diag(betak))))
    
for Wu_k, Wv_k, alphak in zip(model.W_u, model.W_v, model.alpha):
    init_value['alpha'].insert(0, to_np(alphak))  # OG diagonals of each (502,502) CFS Linear Ahat matrix A slice, before low rank
    init_value['W_u'].insert(0, to_np(Wu_k))  # U vectors for each low rank A matrix of each 
    init_value['W_v'].insert(0, to_np(Wv_k))  # V vectors time S singular values for each low rank A matrix
    Wk = torch.mm(Wu_k, Wv_k.T)  # reconstruct each low rank A matrix, (502,5) @ (5,502) = (502,502)
    
    # add the corresponding pre-low rank CFS Linear Ahat OG diagonals back to each reconstructed low_rank matrices
    # insert the A = U @ V.T + D (502,502) to top of Ahat
    Ahat_list.insert(0, to_np(singular_value_norm(Wk + torch.diag(alphak))))

# concatenate them into a vertical np array of the A, B, V matrices in that order
# So Ahat_gd is basically the W of the trained low rank model
Ahat_gd = np.concatenate(Ahat_list, axis = 1)

#Ahat_gd is theta matrix on page 24

np.save('Trained_init' + str(rank_dim) + '.npz', init_value)
np.save('Ahat_gd' + str(rank_dim) + '.npy', Ahat_gd)





############################ CAUSAL CONNECTIVITY MATRIX OF Ahat_gd ##############################
# d = num_neurons
# A = compute_transfer_matrix(Ahat_gd, d)
# plt.figure(figsize=(12,6), dpi=80)
# plt.title('Causal Connectivity Matrix of trained low rank model Ahat_gd')
# plt.imshow(A)

# connection_threshold = 1*np.mean(np.diag(A))
# A_threshold = (A > connection_threshold).astype(float)
# plt.figure(figsize=(12,6), dpi=80)
# plt.title('Causal Connectivity Matrix of trained low rank model Ahat_gd that is above the connection threshold')
# plt.imshow(A > connection_threshold)





######################### LOW RANK TRAINED MODEL USED TO MAKE PREDICTIONS #########################
x_pred = []
x_true = []
u_true = []
r2 = [] 
idx = -1
new_segment = True
segment_pred = []
segment_start = -1
for t in range(train_indices.shape[0]):
    if test_indices[t] == 1: #if it is in test set
        if new_segment:
            segment_pred = []
            new_segment = False
            segment_start = t
            x_past = []
            x_pred.append([])
            x_true.append([])
            u_true.append([])
            idx += 1  # index of segments
        if t < segment_start + ark_order:  
        #if t is within ark_order of segment start, add it to x_past, which should be (2008,) at end of ark_order
            x_past.append(y_session_interp[t,:].copy().flatten()/normalize)
        
        #if not new_segment, and if t is outside current segment
        else:
            z = np.array(x_past).flatten()  # z should be (2008,)
            z = np.concatenate((z,u_session[t-ark_order:t,:].copy().flatten(),np.ones(1)))  # z should now be (4017,)

            # (502, 4017) @ (4017, ) = (502, ) x_next
            x_next = Ahat_gd @ z
            x_past.pop(0)  # pop oldest y_session_interp[t,:] (502,) 
            x_past.append(x_next.copy()) #add the new predicted x_next (502,)
            x_pred[idx].append(x_next.copy())
            x_true[idx].append(y_session_interp[t,:].copy().flatten()/normalize)
            u_true[idx].append(u_session[t,:].copy().flatten())
    else:  #if current t is not part of test set, a new segment should start
        new_segment = True

mse_losses = []
for i in range(len(x_pred)):  # loop through 82 segments
    x_pred[i] = np.array(x_pred[i])  # segment length varies like 66, 109, etc.
    x_true[i] = np.array(x_true[i])
    u_true[i] = np.array(u_true[i])
    mse_losses.append((np.square(x_pred[i] - x_true[i])).mean())
    r2.append(r2_score(x_true[i], x_pred[i]))
print('Trained Low Rank model MSE:', sum(mse_losses)/len(mse_losses))
print('Trained Low Rank model R2:', sum(r2)/len(r2))



############################## TPR/FPR for TRAINED LOW RANK MODEL #################################
tpr = []
fpr = []
results_name = checkpoint_name
thresholds = np.linspace(-2,5,15)

for thresholds_idx in range(len(thresholds)):
    tp_total = 0
    fp_total = 0
    p_total = 0
    n_total = 0

    for neuron in range(u_session.shape[1]):
        output_pred = []
        output_true = []
        for i in range(len(x_true)):
            output_pred.extend(x_pred[i][:,neuron])
            output_true.extend(x_true[i][:,neuron])
        output_pred = np.array(output_pred)
        output_true = np.array(output_true)
    
        mean_threshold = np.median(output_true)
        lower_tail_idx = (output_true < mean_threshold)
        lower_tail_data = output_true[lower_tail_idx]
        lower_tail_std = np.std(lower_tail_data)
        true_spike_threshold = mean_threshold + 6*lower_tail_std
        detect_spike_threshold = mean_threshold + thresholds[thresholds_idx]*lower_tail_std

        predicted_spikes = (output_pred > detect_spike_threshold)
        true_spikes = (output_true > true_spike_threshold)
        tp_total += np.sum(np.logical_and(predicted_spikes,true_spikes))
        fp_total += np.sum(np.logical_and(predicted_spikes,~true_spikes))
        p_total += np.sum(true_spikes)
        n_total += np.sum(~true_spikes)

    tpr.append(tp_total / p_total)
    fpr.append(fp_total / n_total)


tpr_noninput = []
fpr_noninput = []
thresholds = np.linspace(-2,5,15)

for thresholds_idx in range(len(thresholds)):
    tp_total = 0
    fp_total = 0
    p_total = 0
    n_total = 0

    for neuron in range(u_session.shape[1]):
        output_pred = []
        output_true = []
        for i in range(len(x_true)):
            if np.sum(u_true[i][:,neuron]) == 0:
                output_pred.extend(x_pred[i][:,neuron])
                output_true.extend(x_true[i][:,neuron])
        output_pred = np.array(output_pred)
        output_true = np.array(output_true)
    
        mean_threshold = np.median(output_true)
        lower_tail_idx = (output_true < mean_threshold)
        lower_tail_data = output_true[lower_tail_idx]
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
np.save('results/' + results_name + '.npy', results)

results_load = np.load('results/' + results_name + '.npy', allow_pickle=True).item()

# full data linear model vs non-input neurons only

plt.figure(figsize=(8,6), dpi=80)
plt.title(f"{results_name} ROC Curves")
plt.plot(results_load['fpr_noninput'],results_load['tpr_noninput'],label='unexcited-linear')
plt.plot(results_load['fpr'],results_load['tpr'],label='all-linear')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.savefig('results/' + results_name + '_TPR_FPR.pdf')

print(f"the length of x_pred is {len(x_pred)}")
print(f"the length of x_true is {len(x_true)}")
print(f"the length of u_true is {len(u_true)}")




# aurocs_noninput = []
# aurocs = []

results_load = np.load('results/' + results_name + '.npy', allow_pickle=True).item()

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




##################### TRAINED LOW RANK MODEL Performance PLOT ######################

    
neuron = removed_neurons[9]
t_start = 30

win_len = 25

output_pred = []
output_true = []
input_true = []
segment_marker = []

for i in range(t_start, t_start + win_len):
    output_pred.extend(x_pred[i][:,neuron])
    #output_pred.append(np.nan)
    output_true.extend(x_true[i][:,neuron])
    #output_true.append(np.nan)
    input_true.extend(u_true[i][:,neuron])
    #input_true.append(np.nan)
    segment_marker.extend(np.nan*np.zeros(len(x_pred[i][:,neuron])-1))
    segment_marker.extend([0.25,0])
    
plt.figure(figsize=(12,6), dpi=80)
plt.subplot(2,1,1)
plt.plot(output_true,label='truth')
plt.plot(output_pred,label='predicted')
#plt.plot(segment_marker)
plt.title(results_name)
plt.legend()

plt.subplot(2,1,2)
plt.plot(input_true,label='input')
plt.title('Input')
plt.savefig('results/' + results_name + '.pdf')

print(f"length of removed_neurons is {len(removed_neurons)}")
print(f"removed_neurons is {removed_neurons}")



neuron = removed_neurons[8]
length = 50
t_start = 30

output_pred = []
output_true = []
input_true = []
segment_marker = []

for i in range(t_start, t_start + np.min([len(x_true),length])):
    output_pred.extend(x_pred[i][:,neuron])
    #output_pred.append(np.nan)
    output_true.extend(x_true[i][:,neuron])
    #output_true.append(np.nan)
    input_true.extend(u_true[i][:,neuron])
    #input_true.append(np.nan)
    segment_marker.extend(np.nan*np.zeros(len(x_pred[i][:,neuron])-1))
    segment_marker.extend([0.25,0])
    
plt.figure(figsize=(12,6), dpi=80)
plt.subplot(2,1,1)
plt.plot(output_true,label='truth')
plt.plot(output_pred,label='predicted')
#plt.plot(segment_marker)
plt.title(results_name)
plt.legend()

plt.subplot(2,1,2)
plt.plot(input_true,label='input')
plt.title('Input')
plt.savefig('results/' + results_name + '_2' + '.pdf')
plt.show()