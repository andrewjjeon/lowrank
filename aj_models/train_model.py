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

num_neurons = config['num_neurons']
batch_size = config['batch_size']  # how many samples/t's/indices per batch to load in DataLoader

normalize = config['normalize']
epochs = config['epochs']

print(f"hello??? epochs is {epochs}")

learning_rate = config['learning_rate']
lambda_val = config['lambda_val']

train_start = config['ark_order'] + 1
ark_order = config['ark_order']
num_lags = config['ark_order']
init_toggle = config['init_toggle']

Ahat = np.load('Ahat.npy')
u_session = np.load('u_session.npy')
y_session_interp = np.load('y_session_interp.npy')

train_test_split = np.load('train_test_split.npy', allow_pickle=True).item()
train_indices = train_test_split['train_indices']
test_indices = train_test_split['test_indices']


# comment out the init_value blocks if you want to initialize with nothing, instead of Ahat matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

init_value = {}
init_value['alpha'] = []
init_value['W'] = []
init_value['beta'] = []
init_value['B'] = []
for i in range(num_lags):
    # return the diagonals and the removed diagonal matrices of the Ahat slices
    diag_alpha, W = diag_off_diag_extraction(
        Ahat[:, num_neurons * (i) :num_neurons * (i+1)])  # add 4 (502=neuron state of x_next, 0:502, 502:1004, 1004:1506, 1506:2008 --> y_session neuron state at previous states at t=0,...,k)
    diag_beta, B = diag_off_diag_extraction(
        Ahat[:, num_neurons * (i+num_lags) :num_neurons * (num_lags+i+1)])  # add 4 (502=neuron state of x_next, 2008:2510, 2510:3012, 3012:3514, 3514:4016 --> u_session inputs at previous states at t=0,...,k)
    
    # alpha and beta are the diagonals of the original Ahat slices, W and B are the Ahat slices with removed diagonals
    init_value['alpha'].insert(0, torch.tensor(diag_alpha).float().to(device))
    init_value['W'].insert(0, torch.tensor(W).float().to(device))
    
    init_value['beta'].insert(0, torch.tensor(diag_beta).float().to(device))
    init_value['B'].insert(0, torch.tensor(B).float().to(device))

V = Ahat[:, num_neurons * (2*num_lags) :num_neurons * (2*num_lags+1)]

init_value['V'] = torch.squeeze(torch.tensor(V)).float().to(device)




######################## INITIALIZE THE LOW RANK MODEL FROM CLOSED-FORM AHAT #######################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

init_value = {}
init_value['alpha'] = []
init_value['W_u'] = []
init_value['W_v'] = []
init_value['beta'] = []
init_value['B_u'] = []
init_value['B_v'] = []
#num_lags is the lag of s historical time steps, s = 0,...,k-1
for i in range(num_lags):
# The Ahat used here is from the closed-form linear solution

    # calculate the low_rank_svd components for the 4 (502,502) 
    # (502=neuron state of x_next, 0:502, 502:1004, 1004:1506, 1506:2008 previous neuron states y_session 
    diag_alpha, Wu, Wv, W_low = low_rank_svd_components_approximation(  # Wv and Bv are the V vectors multiplied with the singular values
        Ahat[:, num_neurons * (i) :num_neurons * (i+1)], rank_dim)
    
    # calculate the low_rank_svd components for the 4 (502,502) 
    # (502=neuron state of x_next, 2008:2510, 2510:3012, 3012:3514, 3514:4016 previous inputs u_session 
    diag_beta, Bu, Bv, B_low = low_rank_svd_components_approximation(
        Ahat[:, num_neurons * (i+num_lags) :num_neurons * (num_lags+i+1)], rank_dim)
    
    init_value['alpha'].insert(0, torch.tensor(diag_alpha).float().to(device))  # each of these lists should have 4 things in them now
    init_value['W_u'].insert(0, torch.tensor(Wu).float().to(device))
    init_value['W_v'].insert(0, torch.tensor(Wv).float().to(device))
    
    init_value['beta'].insert(0, torch.tensor(diag_beta).float().to(device))
    init_value['B_u'].insert(0, torch.tensor(Bu).float().to(device))
    init_value['B_v'].insert(0, torch.tensor(Bv).float().to(device))

# still only 1 element left for bias term from Closed-Form Ahat, so still (502,1)
V = Ahat[:, num_neurons * (2*num_lags) :num_neurons * (2*num_lags+1)]

init_value['V'] = torch.squeeze(torch.tensor(V)).float().to(device)

with open('init_value.pkl', 'wb') as file:
    pickle.dump(init_value, file)


# init_value = None

############################# TRAIN AND EVALUATE THE LOW-RANK MODEL ##############################


# Generate X and U, state and input data 
# Likely need to start at train_start because its AR-k model
X = torch.Tensor(y_session_interp[train_start:,:]/normalize).to(device)
U = torch.Tensor(u_session[train_start:,:]).to(device)


# Create the dataset and dataloader
dataset = TimeseriesDataset(X, U, num_lags)

# Split the dataset only if they are valid training and test indices (if the indices have 4 consecutive 1's for training and 0's for test)
train_subset, val_subset = split_dataset(dataset, train_indices[train_start:], num_lags)
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)  # 12 length 24000/2000 ~ 22968 training t's
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)  # 4 length 8000/2000 ~ 6340 test t's


# Initialize the model with the low_rank model/matrices we already pre-computed above and stored in init_value
model = LowRankLinearDynamicModel(state_dim, input_dim, rank_dim, num_lags, init_value)

checkpoint_name = 'low_rank_' + str(rank_dim) + '_epochs_' + str(epochs) + '_lr_' + str(learning_rate) + '_lambda_' + str(lambda_val) + '_k_' + str(ark_order) + '_init_value_' + init_toggle
print('checkpoint_name:', checkpoint_name)

# Train the model
plt.figure(figsize=(12, 6), dpi=80)

train_model(model, train_loader, val_loader, epochs = epochs, lr = learning_rate, l2_lambda = lambda_val, checkpoint_name = checkpoint_name)

