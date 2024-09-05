import torch
import torchvision
import matplotlib.pyplot as plt

import os
import yaml
import numpy as np
import pdb

import sys
from lfads import LFADS_Net
from utils import read_data, load_parameters, save_parameters, batchify_random_sample
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier

import scipy.io as spio
from sklearn.cluster import AgglomerativeClustering

# Select device to train LFADS on
device = 'cuda' if torch.cuda.is_available() else 'cpu'; print('Using device: %s'%device)

def to_np(tens):
    return tens.detach().cpu().numpy()

def batchify_random_sample_stim_input(data, stim_input, batch_size, ix=None):
    
    """
    Randomly select sample from data, and turn into a batch of size batch_size to generate multiple samples
    from model to average over
    
    Args:
        data (torch.Tensor) : dataset to randomly select sample from
        batch_size (int) : number of sample repeats
    Optional:
        ix (int) : index to select sample. Randomly generated if None
    
    Returns:
        batch (torch.Tensor) : sample repeated batch
        ix (int) : index of repeated sample
    """
    '''
    if len(data) == 2:
        num_trials = data[0].shape[0]
        if type(ix) is not int:
            ix = np.random.randint(num_trials)
        batch = [data[0][ix].unsqueeze(0).repeat(batch_size, 1, 1), data[1][ix].unsqueeze(0).repeat(batch_size, 1, 1)]
    '''
    num_trials = data.shape[0]
    if type(ix) is not int:
        ix = np.random.randint(num_trials)
    batch = data[ix].unsqueeze(0).repeat(batch_size, 1, 1)
    input_batch = stim_input[ix].unsqueeze(0).repeat(batch_size, 1, 1)
    return batch, input_batch, ix

def logLikelihoodGaussian(x, mu, logvar, mask = None):
    '''
    logLikelihoodGaussian(x, mu, logvar, mask):
    
    Log-likeihood of a real-valued observation given a Gaussian distribution with mean 'mu' 
    and standard deviation 'exp(0.5*logvar), mask out artifacts'
    
    Arguments:
        - x (torch.Tensor): Tensor of size batch-size x time-step x input dimensions
        - mu (torch.Tensor): Tensor of size batch-size x time-step x input dimensions
        - logvar (torch.tensor or torch.Tensor): tensor scalar or Tensor of size batch-size x time-step x input dimensions
        - mask (torch.Tensor): Optional, Tensor of of size batch-size x time-step x input dimensions
    '''
    from math import log,pi
    eps = 1e-5
    if mask is not None:
        # loglikelihood = -0.5*((log(2*pi) + logvar + ((x - mu).pow(2)/(torch.exp(logvar)))) * (1 - mask)).mean()
        loglikelihood_item = (log(2*pi) + logvar + ((x - mu).pow(2)/(torch.exp(logvar))))
        loglikelihood = -0.5*(loglikelihood_item * (1-mask).unsqueeze(1).repeat(1, T_var, 1)).mean()
    else:
        loglikelihood = -0.5*((log(2*pi) + logvar + ((x - mu).pow(2)/(torch.exp(logvar))))).mean()
    return loglikelihood

def split_array(arr, scores, minv, maxv, n_splits = 4):
    # Define intervals
    intervals = np.linspace(minv, maxv, n_splits + 1)

    split_values = []
    split_scores = []
    indexes = []

    for i in range(n_splits):
        condition = (arr >= intervals[i]) & (arr < intervals[i+1])
        
        # Handle the condition where value equals maxv for the last interval
        if i == n_splits - 1:
            condition = (arr >= intervals[i]) & (arr <= intervals[i+1])
        
        split_val = arr[condition]
        split_score = scores[condition]
        idx = np.where(condition)[0]

        split_values.append(split_val)
        split_scores.append(split_score)
        indexes.append(idx)

    return split_values, split_scores, indexes

def compute_covariance_matrix_with_missing_data(data_matrix):
    """
    Compute the covariance matrix from a data matrix with missing data (matrix operations version).

    Parameters:
        data_matrix (numpy.ndarray): The data matrix of shape (n_features, n_observations).
        Missing entries are indicated as np.nan.

    Returns:
        numpy.ndarray: The covariance matrix of shape (n_features, n_features).
    """
    
    n_features, n_observations = data_matrix.shape
    
    # Compute the mean while ignoring NaNs, and subtract to center the data
    mean_vector = np.nanmean(data_matrix, axis=1, keepdims=True)
    centered_data = data_matrix - mean_vector
    # Create masks of missing values (NaNs)
    nan_mask = np.isnan(centered_data)
    # Replace NaNs with zeros in the centered data (for computation)
    centered_data[np.isnan(centered_data)] = 0
    # Initialize an empty array to store the count of valid pairs for each feature combination
    count_matrix = np.zeros((n_features, n_features))
    # Compute the count of valid (non-NaN) observation pairs for each feature combination
    for i in range(n_features):
        count_matrix[i, :] = np.sum(np.logical_not(nan_mask) & np.logical_not(nan_mask[i, :]), axis=1)
    # Compute the covariance matrix
    covariance_matrix = (centered_data @ centered_data.T) / (count_matrix - 1)
    # Handle cases where count_matrix - 1 is zero (to avoid division by zero)
    covariance_matrix[count_matrix - 1 == 0] = np.nan
    
    return covariance_matrix

window_len = 16
T_init = 4
T_eval_start = 4
num_unique_groups = 200
num_epoch = 3000
day = '0201'
data_path = f'results/{day}'
if not os.path.exists(data_path):
    os.mkdir(data_path)
data_day = '0117'
T_var = window_len - T_init - T_eval_start
test_group_size = 'mix'

data = np.load(f'../data/{data_day}_test_random_selection_holdout_{test_group_size}_sim_data.npy', allow_pickle=True).item()

F_train_vis = data['F_data_train']
num_train_trials = data['F_data_train'].shape[0]
num_neurons = data['F_data_train'].shape[2]
input_train_vis = np.zeros((num_train_trials, window_len, num_neurons))
input_train_vis[:,T_init+1:T_init+4,:] = 1
input_id_train_vis = (input_train_vis * np.repeat(data['trial_stim_neuron_ids_train'], repeats = window_len, axis = 1))

F_train_vis_ts = torch.Tensor(F_train_vis).to(device)
input_train_vis_ts = torch.Tensor(input_train_vis).to(device)
input_id_train_vis_ts = torch.Tensor(input_id_train_vis).to(device)

_, num_steps, num_cells = F_train_vis_ts.shape

F_train_vis_ts_reshape = torch.reshape(F_train_vis_ts, (F_train_vis_ts.shape[0] * F_train_vis_ts.shape[1], F_train_vis_ts.shape[2]))
data_logvar_train_null_reshape = torch.log(torch.var(F_train_vis_ts_reshape, dim = 0, keepdim = True))
data_logvar_train_null = data_logvar_train_null_reshape.repeat(F_train_vis_ts.shape[0], F_train_vis_ts.shape[1], 1)
data_mean_train_null = torch.mean(torch.mean(F_train_vis_ts, dim = 1, keepdim = True), dim = 0, keepdim = True).repeat(F_train_vis_ts.shape[0], F_train_vis_ts.shape[1], 1)

F_vis = F_train_vis
trial_ave_list_nan = []
stim_group_ids = np.unique(data['trial_stim_group_ids_train']).astype(int)
seq = data['trial_stim_group_ids_train']

for stim_group_id in list(stim_group_ids):
    stim_trial_ids = np.where(seq==stim_group_id)[0]
    trial_ave = np.mean(F_vis[stim_trial_ids], axis = 0)
    norm_F_trial_nan = (trial_ave - trial_ave.mean(0))/trial_ave.std(0)
    stim_neuron_ids = data['group_permutations_train'][stim_group_id]
    norm_F_trial_nan[:,stim_neuron_ids] = np.nan
    trial_ave_list_nan.append(norm_F_trial_nan)

trial_ave_nan = np.concatenate(trial_ave_list_nan)

covariance_matrix = compute_covariance_matrix_with_missing_data(trial_ave_nan.T)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]
total_variance = np.sum(eigenvalues)
variance_ratio = eigenvalues / total_variance
plt.plot(variance_ratio,'.',alpha=0.2)
plt.xlabel('num of pcs')
plt.ylabel('variance explained')

pc1 = eigenvectors[:,0]

################ load model #####################
def load_model(model_name_1):
    load_path_1 = 'models/optogenetic_simu_data_prediction_photostim/' + model_name_1
    hyperparams_1 = load_parameters(load_path_1+'parameters.yaml')
    save_parameters(hyperparams_1)

    model_1 = LFADS_Net(inputs_dim = num_cells, groups_dim = num_unique_groups, T = num_steps, T_init = T_init, dt = 0.05, device=device,
                     model_hyperparams=hyperparams_1).to(device)
    model_1.load_checkpoint('best', output = '')

    metrics_1 = {}

    sampling = False
    model_1.eval()
    prev_save = model_1.save_variables
    with torch.no_grad():
        model_1.save_variables = True
        model_1(F_train_vis_ts, None, input_id_train_vis_ts, None, None, sampling = sampling)
    model_1.save_variables = prev_save # reset to previous value
    pred_mean_train = model_1.outputs_mean.to(device)
    pred_logvar_train = model_1.outputs_logvar.to(device)

    loglikelihood_model_train = logLikelihoodGaussian(
        F_train_vis_ts[:,T_init + T_eval_start:,:], 
        pred_mean_train[:,T_init + T_eval_start:,:], 
        pred_logvar_train[:,T_init + T_eval_start:,:])
    
    print('loglikelihood_model_train:', loglikelihood_model_train)
    
    loglikelihood_model_train_untarget = logLikelihoodGaussian(
        F_train_vis_ts[:,T_init + T_eval_start:,:], 
        pred_mean_train[:,T_init + T_eval_start:,:], 
        pred_logvar_train[:,T_init + T_eval_start:,:],
        mask = input_id_train_vis_ts[:,T_init+1,:])
    
    print('loglikelihood_model_train_untarget:', loglikelihood_model_train_untarget)

    loglikelihood_null_train = logLikelihoodGaussian(
        F_train_vis_ts[:,T_init + T_eval_start:,:], 
        data_mean_train_null[:,T_init + T_eval_start:,:], 
        data_logvar_train_null[:,T_init + T_eval_start:,:])
    
    print('loglikelihood_null_train:', loglikelihood_null_train)
    
    loglikelihood_null_train_untarget = logLikelihoodGaussian(
        F_train_vis_ts[:,T_init + T_eval_start:,:], 
        data_mean_train_null[:,T_init + T_eval_start:,:], 
        data_logvar_train_null[:,T_init + T_eval_start:,:],
        mask = input_id_train_vis_ts[:,T_init+1,:])
    print('loglikelihood_null_train_untarget:', loglikelihood_null_train_untarget)
    
    print('variance_explained_test:', (1 - loglikelihood_model_train/loglikelihood_null_train).detach().cpu().numpy())
    print('variance_explained_test_untarget:', (1 - loglikelihood_model_train_untarget/loglikelihood_null_train_untarget).detach().cpu().numpy())

    metrics_1['variance_explained_test'] = (1 - loglikelihood_model_train/loglikelihood_null_train).detach().cpu().numpy()
    metrics_1['variance_explained_test_untarget'] = (1 - loglikelihood_model_train_untarget/loglikelihood_null_train_untarget).detach().cpu().numpy()

    return model_1, metrics_1

def generate(model, stim_input, batch_size, inputs_dim, T_init, T, g0_sampling, output_sampling, num_groups = 1, noise_scale = 0.1):
    '''
    Generates output.
    '''
    g0_mean = model.g0_mean[:batch_size].repeat([num_groups, 1])
    g0_logvar = model.g0_logvar[:batch_size].repeat([num_groups, 1])
    
    if g0_sampling:
        g = g0_mean + torch.randn_like(g0_logvar).to(device)*torch.exp(0.5*g0_logvar) * noise_scale
    else:
        g = g0_mean
        
    outputs  = torch.zeros(batch_size * num_groups, T, inputs_dim).to(device)
    for t in range(T_init, T):
        u_input = stim_input[:, t, :]
        # Update generator
        g = torch.clamp(model.gru_generator(u_input, g), min=0.0, max=model.clip_val)
        # Generate output Gaussain mu from generator state
        mu = model.fc_mu(g)
        # Generate output Gaussain logvar from generator state
        logvar = model.fc_logvar(g)
        if output_sampling:
            outputs[:, t]   = mu + torch.randn_like(logvar).to(device) * torch.exp(0.5*logvar) * noise_scale
        else:
            outputs[:, t]   = mu
    return outputs

def active_learning(sparsity_lamda = 500.0, num_stim_groups = 300, similarity_lamda = 10.0, sparsity_constraint = 'varied', temp = 10.0, use_pca = False, num_pcs = 5):
    
    num_epoches = 600
    decay_steps = 30
    decay_factor = 0.1
    energy_scale = 10.0
    lr = 1e-1
    use_energy = False
    step_size = 0.5
    T = 16
    g0_sampling = False
    output_sampling = False
    pca_tensor = torch.FloatTensor(eigenvectors[:,:num_pcs]).to(device)
    active_iters = 1
    batch_size = 10

    inputs_dim = input_id_train_vis_ts.shape[2]
    input_t_define = torch.zeros(batch_size * num_stim_groups, T, inputs_dim)
    input_t_define[:,5:8,:] = 1.0
    input_t_define = input_t_define.to(device)

    active_mismatches = []
    active_pca_mismatches = []
    active_inputs = []
    active_input_similarity = []

    for iteration in range(active_iters):

        print(f'iter {iteration}: learn optimal input')
        losses = []
        mismatches = []
        pca_mismatches = []
        sparsity_losses = []
        input_similarity_losses = []
        epochs = []

        optimized_input = torch.randn((num_stim_groups, 1, inputs_dim), requires_grad=True, device=device)

        if not use_energy:
            optimizer = torch.optim.Adam([optimized_input], lr=lr)
        for epoch in range(num_epoches):
            optimized_input_repeat = optimized_input.repeat(batch_size, 1, 1)
            output_mean_0 = generate(model_0, 
                                    input_t_define * torch.sigmoid(temp * optimized_input_repeat), 
                                    batch_size, 
                                    inputs_dim, 
                                    T_init, 
                                    T,
                                    num_groups = num_stim_groups,
                                    g0_sampling = g0_sampling,
                                    output_sampling = output_sampling)

            output_mean_1 = generate(model_1, 
                                    input_t_define * torch.sigmoid(temp * optimized_input_repeat), 
                                    batch_size, 
                                    inputs_dim, 
                                    T_init, 
                                    T,
                                    num_groups = num_stim_groups,
                                    g0_sampling = g0_sampling,
                                    output_sampling = output_sampling)

            # Maximize A (therefore minimize -A)
            pca_mismatch = torch.norm(torch.matmul(output_mean_0 - output_mean_1, pca_tensor), p = 2, dim = (1,2)).mean(0)
            mismatch = torch.norm(output_mean_0 - output_mean_1, p = 2, dim = (1,2)).mean(0)
            sparsity_loss = torch.mean(torch.sigmoid(temp * optimized_input))

            optimized_input_test = torch.sigmoid(temp * optimized_input).squeeze(1)
            input_similarity_before_norm = torch.mm(optimized_input_test, optimized_input_test.t()).fill_diagonal_(0)
            # Calculate the norm of each row vector in optimized_input_test
            norms = optimized_input_test.norm(dim=1)
            # Calculate outer product of norms to get a matrix of shape (B, B)
            norm_matrix = torch.outer(norms, norms)
            # Normalize the result matrix
            input_similarity_matrix = input_similarity_before_norm / norm_matrix
            input_similarity = torch.mean(input_similarity_matrix)

            if sparsity_constraint == 'varied':
                sparsity_lamda_range = torch.arange(start = 0, end = sparsity_lamda, step = sparsity_lamda/num_stim_groups).to(device)
                weighted_sparsity_loss = torch.mean(torch.sigmoid(temp * optimized_input) * sparsity_lamda_range[:,None,None])
            else:
                weighted_sparsity_loss = sparsity_loss * sparsity_lamda
            if use_pca:
                loss = - pca_mismatch + weighted_sparsity_loss + similarity_lamda * input_similarity
            else:
                loss = - mismatch + weighted_sparsity_loss + similarity_lamda * input_similarity
            loss.backward()
            if use_energy:
                if epoch % decay_steps == 0 and i > 0:
                    step_size *= decay_factor
                with torch.no_grad():
                    optimized_input -= energy_scale * step_size * optimized_input.grad
                    optimized_input += step_size * 2 * torch.randn_like(optimized_input).to(device)
            else:
                optimizer.step()
                optimizer.zero_grad()
            epochs.append(epoch)
            losses.append(loss.item())
            mismatches.append(mismatch.item())
            pca_mismatches.append(pca_mismatch.item())
            sparsity_losses.append(sparsity_loss.item())
            input_similarity_losses.append(input_similarity.item())

        pca_mismatch_groups = torch.norm(torch.matmul(output_mean_0 - output_mean_1, pca_tensor), p = 2, dim =(1,2)).view(batch_size, num_stim_groups).mean(0)
        mismatch_groups = torch.norm(output_mean_0 - output_mean_1, p = 2, dim =(1,2)).view(batch_size, num_stim_groups).mean(0)

        optimized_input_test_discrete = (torch.sigmoid(temp * optimized_input) > 0.5) * 1.0
        print('num_stim_neurons per group:', torch.sum(optimized_input_test_discrete)/num_stim_groups)
        stim_neuron_ids = np.where(to_np(optimized_input_test_discrete[:,0])!= 0)[1]
        print('stim neuron ids:', stim_neuron_ids)

        rand_input = torch.rand(num_stim_groups, 1, inputs_dim).to(device)
        if sparsity_constraint == 'varied':
            sparsity_loss_group = torch.mean(torch.sigmoid(temp * optimized_input), dim = 2)
            ones_indices = (rand_input <= sparsity_loss_group[:,:,None])
            zero_indices = (rand_input > sparsity_loss_group[:,:,None])
            rand_input[ones_indices] = 1
            rand_input[zero_indices] = 0
        else:
            ones_indices = (rand_input <= sparsity_loss)
            zero_indices = (rand_input > sparsity_loss)
            rand_input[ones_indices] = 1
            rand_input[zero_indices] = 0

        rand_input_repeat = rand_input.repeat(batch_size, 1, 1)
        output_mean_0 = generate(model_0, 
                                input_t_define * rand_input_repeat, 
                                batch_size, 
                                inputs_dim, 
                                T_init, 
                                T,
                                num_groups = num_stim_groups,
                                g0_sampling = g0_sampling,
                                output_sampling = output_sampling)

        output_mean_1 = generate(model_1, 
                                input_t_define * rand_input_repeat,
                                batch_size, 
                                inputs_dim, 
                                T_init, 
                                T,
                                num_groups = num_stim_groups,
                                g0_sampling = g0_sampling,
                                output_sampling = output_sampling)

        rand_pca_mismatch = torch.norm(torch.matmul(output_mean_0 - output_mean_1, pca_tensor), p = 2, dim = (1,2)).mean(0)
        rand_mismatch = torch.norm(output_mean_0 - output_mean_1, p = 2, dim = (1,2)).mean(0)
        rand_pca_mismatch_groups = torch.norm(torch.matmul(output_mean_0 - output_mean_1, pca_tensor), p = 2, dim =(1,2)).view(batch_size, num_stim_groups).mean(0)
        rand_mismatch_groups = torch.norm(output_mean_0 - output_mean_1, p = 2, dim =(1,2)).view(batch_size, num_stim_groups).mean(0)

        fig, axes = plt.subplots(2,3,figsize=(15,10))
        axes[0,0].plot(np.asarray(epochs), np.asarray(losses))
        axes[0,0].set_xlabel('epoch')
        axes[0,0].set_ylabel('loss')

        axes[0,1].plot(np.asarray(epochs), np.asarray(mismatches))
        axes[0,1].plot(np.asarray(epochs), np.ones_like(mismatches) * rand_mismatch.item())
        axes[0,1].set_xlabel('epoch')
        axes[0,1].set_ylabel('mismatches')
        axes[0,1].legend(['optimized', 'rand'])

        axes[0,2].plot(np.asarray(epochs), np.asarray(pca_mismatches))
        axes[0,2].plot(np.asarray(epochs), np.ones_like(pca_mismatches) * rand_pca_mismatch.item())
        axes[0,2].set_xlabel('epoch')
        axes[0,2].set_ylabel('pca_mismatches')
        axes[0,2].legend(['optimized', 'rand'])

        axes[1,0].plot(np.asarray(epochs), np.asarray(sparsity_losses))
        axes[1,0].plot(np.asarray(epochs), np.ones_like(sparsity_losses)*to_np(torch.sum(rand_input)/torch.numel(rand_input)))
        axes[1,0].set_xlabel('epoch')
        axes[1,0].set_ylabel('sparsity_loss')

        axes[1,1].plot(np.asarray(epochs), np.asarray(input_similarity_losses))
        axes[1,1].set_xlabel('epoch')
        axes[1,1].set_ylabel('input_similarity_losses')
        fig.savefig(f'results/{day}/day_{day}_active_learning_optimize_loss_sparse_{sparsity_lamda}_sparsity_constraint_{sparsity_constraint}_temp_{temp}_diverse_{similarity_lamda}.png')

        plt.figure()
        plt.imshow(to_np(optimized_input_test_discrete[:,0]).astype(int))
        plt.xlabel('neuron_index')
        plt.ylabel('group_index')
        plt.title('binary stimulation')
        plt.colorbar()
        plt.savefig(f'results/{day}/day_{day}_optimized_input_binary_sparse_{sparsity_lamda}_sparsity_constraint_{sparsity_constraint}_temp_{temp}_diverse_{similarity_lamda}.png')

        plt.figure()
        plt.imshow(to_np(torch.sigmoid(temp * optimized_input))[:,0])
        plt.xlabel('neuron_index')
        plt.ylabel('group_index')
        plt.title('continous stimulation')
        plt.colorbar()
        plt.savefig(f'results/{day}/day_{day}_optimized_input_sigmoid_sparse_{sparsity_lamda}_sparsity_constraint_{sparsity_constraint}_temp_{temp}_diverse_{similarity_lamda}.png')

        plt.figure()
        plt.plot(np.sum(to_np(optimized_input_test_discrete[:,0]), axis = 1), 'o')
        plt.xlabel('group_index')
        plt.ylabel('num of target neurons')
        plt.title('num target neuron distribution')
        plt.savefig(f'results/{day}/day_{day}_optimized_input_num_target_neurons_distribution_sparse_{sparsity_lamda}_sparsity_constraint_{sparsity_constraint}_temp_{temp}_diverse_{similarity_lamda}.png')

        plt.figure()
        plt.imshow(to_np(input_similarity_matrix), vmin = 0, vmax = 1)
        plt.xlabel('group_index')
        plt.ylabel('group_index')
        plt.title('input similarity')
        plt.colorbar()
        print('input similarity:', to_np(input_similarity))
        plt.savefig(f'results/{day}/day_{day}_optimized_input_similarity_sparse_{sparsity_lamda}_sparsity_constraint_{sparsity_constraint}_temp_{temp}_diverse_{similarity_lamda}.png')

        plt.figure()
        sort_group_indexes = np.argsort(to_np(pca_mismatch_groups))[::-1]
        plt.plot(to_np(mismatch_groups)[sort_group_indexes],'.')
        plt.plot(to_np(pca_mismatch_groups)[sort_group_indexes],'.')

        rand_sort_group_indexes = np.argsort(to_np(rand_pca_mismatch_groups))[::-1]
        plt.plot(to_np(rand_mismatch_groups)[rand_sort_group_indexes],'.')
        plt.plot(to_np(rand_pca_mismatch_groups)[rand_sort_group_indexes],'.')

        plt.xlabel('group_index (sorted)')
        plt.ylabel('mismatch')
        plt.legend(['mismatch', 'pca_mismatch (sorted)', 'rand_mismatch', 'rand_pca_mismatch (sorted)'])
        plt.title('group_index & mismatch (optimized v.s. rand)')
        plt.savefig(f'results/{day}/day_{day}_optimized_vs_random_mismatch_sorted_sparse_{sparsity_lamda}_sparsity_constraint_{sparsity_constraint}_temp_{temp}_diverse_{similarity_lamda}.png')

        plt.figure()
        plt.plot(to_np(mismatch_groups),'.')
        plt.plot(to_np(pca_mismatch_groups),'.')

        plt.plot(to_np(rand_mismatch_groups),'.')
        plt.plot(to_np(rand_pca_mismatch_groups),'.')

        plt.xlabel('group_index (w/o sorted)')
        plt.ylabel('mismatch')
        plt.legend(['mismatch', 'pca_mismatch', 'rand_mismatch', 'rand_pca_mismatch'])
        plt.title('group_index & mismatch (optimized v.s. rand)')
        plt.savefig(f'results/{day}/day_{day}_optimized_vs_random_mismatch_sparse_{sparsity_lamda}_sparsity_constraint_{sparsity_constraint}_temp_{temp}_diverse_{similarity_lamda}.png')

        active_mismatches.append(mismatch.item())
        active_pca_mismatches.append(pca_mismatch.item())
        active_inputs.append(optimized_input_test.detach())
        active_input_similarity.append(input_similarity.item())

    outputs = {'stim_inputs': to_np(active_inputs[0]),
               'sparsity': sparsity_loss.item(),
               'similarity': input_similarity.item(),
               'pca_mismatch': pca_mismatch.item(),
               'mismatch': mismatch.item(),
               'rand_inputs': to_np(rand_input),
               'rand_pca_mismatch': rand_pca_mismatch.item(),
               'rand_mismatch': rand_mismatch.item()}
    print(outputs)

    np.save(f'../data/simu_inputs_{day}_pca_{use_pca}_sparse_{sparsity_lamda}_sparsity_constraint_{sparsity_constraint}_temp_{temp}_similarity_{similarity_lamda}_num_pcs_{num_pcs}.npy', outputs)

# before active learning
group_size = 'mix'
num_groups = '200'
fname_0 = f'photostim_random_learning_g_1024_01_17_stim_group_{group_size}_num_groups_{num_groups}_rand_day_0_rand_0/'
fname_1 = f'photostim_random_learning_g_1024_01_17_stim_group_{group_size}_num_groups_{num_groups}_rand_day_0_rand_1/'

model_0, metrics_0 = load_model(model_name_1 = fname_0)
model_1, metrics_1 = load_model(model_name_1 = fname_1)

active_learning(sparsity_lamda = 70.0,
                similarity_lamda = 100.0,
                sparsity_constraint = 'varied',
                num_stim_groups = 300,
                temp = 1.0,
                use_pca = False,
                num_pcs = num_neurons)