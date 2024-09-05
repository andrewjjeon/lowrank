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
from sklearn.metrics import r2_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
#
#
#
##################### FULL RANK AND LOW RANK MODEL ######################


# recreate/filter the indices to make sure the k intervals are respected

def to_np(tens):
    return tens.detach().cpu().numpy()

# NOTE: This function is recreating/filtering train_indices and val_indices, it only adds i from input, IF i and num_lags i's after are ALL 1 (train) or ALL 0 (val) respectively
def filter_indices_with_lags(train_data_index, num_lags):
    train_indices = []
    val_indices = []
    # train_data_index = train_indices[train_start:]
    # NOTE: minus num_lags so our indexing doesn't break at the end
    # looping through train_indices[5:29304], 5=train_start from start, 4=num_lags from last
    for i in range(len(train_data_index) - num_lags):

        # if current i and the num_lags next i's are ALL 1, add to train_indices
        if all(train_data_index[i + j] == 1 for j in range(num_lags + 1)):
            train_indices.append(i)
        # else if current i and the num_lags next i's are ALL 0, add to val_indices
        elif all(train_data_index[i + j] == 0 for j in range(num_lags + 1)):
            val_indices.append(i)
    
    return train_indices, val_indices

def split_dataset(dataset, train_data_index, num_lags):
    train_indices, val_indices = filter_indices_with_lags(train_data_index, num_lags)
    
    # Create subsets for training and validation
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    return train_subset, val_subset




class TimeseriesDataset(Dataset):

    # X is basically torch.tensor(y_session), U is basically torch.tensor(u_session)
    def __init__(self, X, U, num_lags):
        self.X = X
        self.U = U 
        self.num_lags = num_lags

    def __len__(self):
        # The dataset length is reduced by num_lags due to the dependency on previous data points
        return len(self.X) - self.num_lags

# NOTE: This function is returning 4 index history of X and U as well as the prediction point X
    def __getitem__(self, index):
        # return slices of X and U of [index:index + num_lags]
        X_history = [self.X[index + i] for i in range(self.num_lags)]
        U_history = [self.U[index + i] for i in range(self.num_lags)]

        # X_next is the prediction point
        X_next = self.X[index + self.num_lags]
        return X_history, U_history, X_next

# NOTE: This function normalizes matrix such that the largest singular value becomes 2
def singular_value_norm(matrix):
    norm_val = torch.linalg.norm(matrix, 2)  # norm_val = largest singular value of matrix
    if norm_val > 2: 
        matrix = 2 * matrix / norm_val  # normalized/scaled such that the largest singular values becomes 2
    return matrix


class LinearDynamicModel(nn.Module):  # inherit from nn.Module which is base class for all neural network in pytorch
    def __init__(self, state_dim, input_dim, num_lags, init_value = None):
        
        super(LinearDynamicModel, self).__init__()  # constructor of nn.Module, gives functionalities of nn.Module to LinearDynamicModel
        
        # if doesn't exist yet, create a new parameter list
        if init_value is None:
            # Create diagonal matrices for alpha and beta, one for each lag

            # nn.ParameterList is like Python list, but tensors that are nn.Parameter are visible by all Module methods and autograd will work
            # torch.randn(state_dim) creates tensor of len(state_dim) of random numbers from normal distribution (0, 1)
            # alpha and beta are lists of parameters with num_lags=4 tensors of size state_dim=502
            self.alpha = nn.ParameterList([nn.Parameter(torch.randn(state_dim)) for _ in range(num_lags)])
            self.beta = nn.ParameterList([nn.Parameter(torch.randn(state_dim)) for _ in range(num_lags)])
            
            # W is matrix A and B is matrix B in the paper
            # W and B are lists of parameters with num_lags=4 tensors of size (state_dim, state_dim) and (state_dim, input_dim)
            self.W = nn.ParameterList([nn.Parameter(torch.randn(state_dim, state_dim)) for _ in range(num_lags)])
            self.B = nn.ParameterList([nn.Parameter(torch.randn(state_dim, input_dim)) for _ in range(num_lags)]) # this is full-rank, linear model so i think state_dim=input_dim

            # V is a parameter tensor of size state_dim
            self.V = nn.Parameter(torch.randn(state_dim))

        # if already exists, create the parameter list by pulling from the existing dictionary init_value
        else:
            # init_value is a {} dictionary of [] lists
            self.alpha = nn.ParameterList([nn.Parameter(init_value['alpha'][i]) for i in range(num_lags)])
            self.beta = nn.ParameterList([nn.Parameter(init_value['beta'][i]) for i in range(num_lags)])

            self.W = nn.ParameterList([nn.Parameter(init_value['W'][i]) for i in range(num_lags)])
            self.B = nn.ParameterList([nn.Parameter(init_value['B'][i]) for _ in range(num_lags)])

            self.V = nn.Parameter(init_value['V'])
    

    def forward(self, X_history, U_history):
        # initialize tensor of 0s of same size as first X_history tensor so we would expect X_next to be size (502)
        X_next = torch.zeros_like(X_history[0])

        #  self.W is ParameterList of 4 tensors of size (502, 502)
        #  self.alpha is ParameterList of 4 tensors of size (502)
        #  X_history is python list of size (4, 502)
        for W_k, alpha_k, X_k in zip(self.W, self.alpha, X_history):
            # W_k (502, 502)
            # alpha_k (502)
            # X_k (502)

            X_k = X_k.unsqueeze(-1)
            alpha_diag_k = torch.diag(alpha_k)

            # unsqueeze X_k so it now has shape (502, 1), add extra dimension
            # torch.diag(alpha_k) so alpha_diag_k has shape (502, 502)

            # compute contribution of state X_k to state X_next
            # matrices A and B correspond to W + diag(alpha) and B + diag(beta)
            # (502, 502) @ (502, 1) = (502,1).squeeze(-1) = (502)
            contribution = torch.matmul(singular_value_norm(W_k + alpha_diag_k), X_k).squeeze(-1)

            # X_next is (502) of 0's so add contribution to it
            # X_next now has the contribution of num_lags previous states
            X_next += contribution

        #  self.B is ParameterList of 4 tensors of size (502, input_dim)
        #  self.beta is ParameterList of 4 tensors of size (502)
        #  U_history is python list of size (4, 502)
        for B_k, beta_k, U_k in zip(self.B, self.beta, U_history):
            U_k = U_k.unsqueeze(-1)
            beta_diag_k = torch.diag(beta_k)
            # B_k (502, input_dim=502), full rank model so state_dim = input_dim
            # U_k (502, 1)
            # beta_diag_k (502, 502)

            # compute contribution of input U_k to next state X-next
            # (502,502) @ (502,1) = (502,1).squeeze(-1) = (502)
            contribution = torch.matmul(singular_value_norm(B_k + beta_diag_k), U_k).squeeze(-1)

            # X_next now has contribution of num_lags previous states AND num_lags previous inputs
            # X_next is still (502)
            X_next += contribution

        # X_next is still (502)
        X_next += self.V[None, :]
        return X_next
    
class LowRankLinearDynamicModel(nn.Module):  # inherit from nn.Module which is base class for all neural network in pytorch
    def __init__(self, state_dim, input_dim, rank_dim, num_lags, init_value = None):
        super(LowRankLinearDynamicModel, self).__init__()  # constructor of nn.Module, gives functionalities of nn.Module to LinearDynamicModel
        
        if init_value is None:
            # self.alpha and self.beta are both ParameterList of 4 Parameters each a tensor of size (502)
            self.alpha = nn.ParameterList([nn.Parameter(torch.randn(state_dim)) for _ in range(num_lags)])
            self.beta = nn.ParameterList([nn.Parameter(torch.randn(state_dim)) for _ in range(num_lags)])

            # self.W_u and self.W_v are both ParameterList of 4 Parameters each a tensor of size (502, 5)
            self.W_u = nn.ParameterList([nn.Parameter(torch.randn(state_dim, rank_dim)) for _ in range(num_lags)])
            self.W_v = nn.ParameterList([nn.Parameter(torch.randn(state_dim, rank_dim)) for _ in range(num_lags)])

            # self.B_u and self.B_v are both ParameterList of 4 Parameters each a tensor of size (502, 5)
            self.B_u = nn.ParameterList([nn.Parameter(torch.randn(state_dim, rank_dim)) for _ in range(num_lags)])
            self.B_v = nn.ParameterList([nn.Parameter(torch.randn(state_dim, rank_dim)) for _ in range(num_lags)])

            # self.V is a Parameter tensor of size (502)
            self.V = nn.Parameter(torch.randn(state_dim))
        else:
            self.alpha = nn.ParameterList([nn.Parameter(init_value['alpha'][i]) for i in range(num_lags)])
            self.beta = nn.ParameterList([nn.Parameter(init_value['beta'][i]) for i in range(num_lags)])

            self.W_u = nn.ParameterList([nn.Parameter(init_value['W_u'][i]) for i in range(num_lags)])
            self.W_v = nn.ParameterList([nn.Parameter(init_value['W_v'][i]) for i in range(num_lags)])

            self.B_u = nn.ParameterList([nn.Parameter(init_value['B_u'][i]) for i in range(num_lags)])
            self.B_v = nn.ParameterList([nn.Parameter(init_value['B_v'][i]) for i in range(num_lags)])

            self.V = nn.Parameter(init_value['V'])
        
    def forward(self, X_history, U_history):
        X_next = torch.zeros_like(X_history[0])  # (502)
        for W_u_k, W_v_k, alpha_k, X_k in zip(self.W_u, self.W_v, self.alpha, X_history):
            X_k = X_k.unsqueeze(-1)  
            alpha_diag_k = torch.diag(alpha_k)
            # W_u_k and W_v_k(502, 35) low rank approx. of matrix A
            # alpha_diag_k (502, 502) original diagonals of each of the 4 (502,502) in Ahat ~ y_session[t:t+4]
            # X_k (502, 1) each of the 4 previous states

            # U_A @ V_A.T (502, 502)
            W_k = torch.mm(W_u_k, W_v_k.T)  # reconstruct each of 4 A matrix 
            
            # A_s = U_A @ V_A.T + D_A
            # (502, 502) @ (502, 1) = (502, 1).squeeze(-1) = (502)
            contribution = torch.matmul(singular_value_norm(W_k + alpha_diag_k), X_k).squeeze(-1)  # Shape returns to (batch_size, state_dim)
            X_next += contribution  # add each of the 4 contributions from the previous states to x_next (502,)

        for B_u_k, B_v_k, beta_k, U_k in zip(self.B_u, self.B_v, self.beta, U_history):
            U_k = U_k.unsqueeze(-1)
            beta_diag_k = torch.diag(beta_k)
            # B_u_k and B_v_k (502, 35) low rank approx. of matrix B
            # beta_diag_k (502, 502) (502, 502) original diagonals of each of the 4 (502,502) in Ahat ~ u_session[t:t+4]
            # U_k (502, 1) each of the 4 previous inputs

            # U_B @ V_B.T (502, 502)
            B_k = torch.mm(B_u_k, B_v_k.T)  # reconstruct each of 4 B matrix 
            
            # B_s = U_B @ V_B.T + D_B
            # (502, 502) @ (502, 1) = (502, 1).squeeze(-1) = (502)
            contribution = torch.matmul(singular_value_norm(B_k + beta_diag_k), U_k).squeeze(-1)
            X_next += contribution  # add each of the 4 contributions from the previous inputs to x_next (502,)

        # X_next is (502,) of all contributions from each of the previous 4 states and inputs    
        X_next += self.V[None, :]  # account for bias term
        return X_next

def train_model(model, train_loader, val_loader, epochs=100, lr=0.01, clip_value=1.0, l2_lambda=0.01, step_size=50, gamma=0.5, checkpoint_name = 'linear_35'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Decays the lr of each parameter by a factor of gamma at every step_size
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    
    # Lists to track loss
    train_losses = []
    reg_losses = []
    val_losses = []
    l2_penalty_losses = []
    
    current_lr = lr

    for epoch in range(epochs):
        # Training Phase
        model.train()  # set model to training mode
        total_train_loss = 0
        reg_train_loss = 0
        total_l2_penalty = 0

        for X_history, U_history, X_next in train_loader:  # loop through 12 train batches in train_loader
            # print(f"X_history[0] shape is {len(X_history[0])} there are 4 of these")
            # print(f"U_history[0] shape is {len(U_history[0])} there are 4 of these")
            # print(f"X_next shape is {len(X_next)}")
            
            optimizer.zero_grad()  # zero out gradients for each batch
            predictions = model(X_history, U_history)  # this is X_next (502) from forward method
            loss = criterion(predictions, X_next)  # take loss betwen X_next (502) predicted and X_next true
            # print(f"pre_L2_loss: {loss}")
            
            total_train_loss += loss.item()  # MSE train before regularization
            
            # Compute the L2 penalty for each parameter
            l2_norm = torch.tensor(0.).to(device)
            
            for param in model.W_u:  # each param is (state_dim, rank_dim)
                l2_norm += torch.norm(param,p=2)
            # print(f"l2_penalty is {l2_penalty}")
            for param in model.W_v:
                l2_norm += torch.norm(param,p=2)
            for param in model.B_u:
                l2_norm += torch.norm(param,p=2)
            for param in model.B_v:
                l2_norm += torch.norm(param,p=2)
            for param in model.alpha:  # each param here is (state_dim,)
                l2_norm += torch.norm(param,p=2)
            for param in model.beta:
                l2_norm += torch.norm(param,p=2)
            
            # Add the L2 penalty to the original loss
            l2_penalty = l2_lambda * l2_norm
            loss += l2_penalty
            
            reg_train_loss += loss.item()

            # print(f"post_L2_loss: {loss}")

            total_l2_penalty += l2_penalty.item()  # L2 norm after times lambda

            # accumulates dloss/dx for every parameter x into x.grad for every parameter x
            # x.grad += dloss/dx
            loss.backward()

            # x += -lr * x.grad, update parameters with gradients
            optimizer.step()
        
        # Validation Phase
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for X_history, U_history, X_next in val_loader:  # loop through 4 val batches of 2000 in val_loader
                predictions = model(X_history, U_history)
                loss = criterion(predictions, X_next)
                total_val_loss += loss.item()

        # Logging average training and validation loss, and L2 penalty
        # TODO: how does __getitem__ work?
        train_loss = total_train_loss / len(train_loader)
        reg_loss = reg_train_loss / len(train_loader)
        val_loss = total_val_loss / len(val_loader)
        l2_penalty_loss = total_l2_penalty / len(train_loader)
        
        train_losses.append(train_loss)
        reg_losses.append(reg_loss)
        val_losses.append(val_loss)
        l2_penalty_losses.append(l2_penalty_loss)
        
        print(f'Epoch {epoch}: Train Loss = {train_loss}, Reg Loss = {reg_loss}, Val Loss = {val_loss}, L2 Penalty term = {l2_penalty_loss}, LR = {current_lr}')

        # Checkpointing based on minimal validation loss across all epochs
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path("checkpoints").mkdir(exist_ok=True)
            checkpoint_path = f'checkpoints/model_best_' + checkpoint_name + '.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch} with Val Loss: {val_loss:.4f}")
    
    print("am i valid")
    # Plotting the training and validation losses
    plt.figure(figsize=(12, 6), dpi=80)
    plt.plot(train_losses, label='Train Loss')  #if no x provided, plots against indices
    plt.plot(val_losses, label='Val Loss')
    plt.plot(reg_losses, label='Reg Train Loss')
    plt.title(f"Train, Reg Train and Val Loss w/ L2 Reg of lambda {l2_lambda}")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('C:/Users/andre/Desktop/active_neuron/plots/' + checkpoint_name + '_losses.png')
    
    # Plotting the L2 penalties
    plt.figure(figsize=(12, 6), dpi=80)
    plt.plot(l2_penalty_losses, label='L2 Penalty terms', color='red')
    plt.title(f"L2 Penalty Over Epochs of lambda {l2_lambda}")
    plt.xlabel('Epochs')
    plt.ylabel('L2 Penalty')
    plt.legend()
    plt.grid(True)
    plt.savefig('C:/Users/andre/Desktop/active_neuron/plots/' + checkpoint_name + '_penalties.png')
    plt.show()






# NOTE: This function returns the original diagonal elements of A, top r U vectors, top r singular values @ top r Vt vectors transposed, low-rank approximation of matrix
# TODO: ???Why are we subtracting the original diagonal
def low_rank_svd_components_approximation(A, r):
    # Perform Singular Value Decomposition
    A_r = A - np.diag(np.diag(A))  #A_r is A with diagonal elements 0'ed out
    U, s, Vt = np.linalg.svd(A_r, full_matrices=False)
    # Keep only the top r singular values (and corresponding vectors)
    Ur = U[:, :r]  # top r U vectors
    Sr = np.diag(s[:r])  # vector of the top r singular values
    Vtr = Vt[:r, :]  # top r V vectors

    # Reconstruct the low-rank approximation of the matrix
    Ar = np.dot(Ur, np.dot(Sr, Vtr))
    return np.diag(A), Ur, np.dot(Sr, Vtr).T, Ar

def diag_off_diag_extraction(A):
    # return original diagonal elements and the original matrix - diagonal elements
    A_r = A - np.diag(np.diag(A))
    return np.diag(A), A_r


# Causal Connectivity Matrix

def compute_transfer_matrix(Ahat, d):
    k = 4
    avg_connect_ark = np.zeros((d,d))  # initialize the causal connectivity matrix (502,502) of 0's
    rollout_len = 1000  #1000 is enough for steady-state approximation
    params = []
    A_params = []
    B_params = []
    for i in range(k):
        params.append(np.zeros((d,d)))  # initialize list of 4 (502,502) matrix in params

        A_params.insert(0,Ahat[:,i*d:(i+1)*d])  # add 4 (502=neuron state of x_next, 0:502, 502:1004, 1004:1506, 1506:2008 --> y_session neuron state at previou states at t=0,...,k) check yellow paper for visualization
        B_params.insert(0,Ahat[:,d*k+i*d:d*k+(i+1)*d])  # add 4 (502=neuron state of x_next, 2008:2510, 2510:3012, 3012:3514, 3514:4016 --> u_session inputs at previous states at t=0,...,k) check yellow paper for visualization
    
    # walking through the causal connectivity math in 3.2 and this function would be helpful
    # Andrew Wagenmaker wrote this function
    
    for t in range(rollout_len):
        param_new = np.zeros((d,d))

        # param_new would be 0 for first 4 iterations

        for i in range(k):
            param_new += A_params[i] @ params[i]

        # if t is 0,1,2,3, add the u_session data from B_params to param_new
        # so we are adding the u_session data from B_params, 4 times to param_new
        if t <= k-1:
            param_new += B_params[t]
        
        # so there are 4 (502,502) matrices in params, now remove the last one
        params = params[:-1]

        # add param_new (accumulated u_session data for 0:k) to beginning of params
        params.insert(0,param_new)
        avg_connect_ark += params[0]
    return avg_connect_ark

#plot singular values of diagonals removed np.diag(A) A and regular A
def plot_rank_svd(A):
    A2 = A - np.diag(np.diag(A))
    U_true,S_true,V_true = np.linalg.svd(A2)

    plt.plot(S_true,label='A minus diag')
    plt.xlabel('singular value index')
    plt.ylabel('singular value')

    U_true,S_true,V_true = np.linalg.svd(A)
    plt.plot(S_true,label='full A')
    plt.xlabel('singular value index')
    plt.ylabel('singular value')

    plt.legend()
    plt.show()