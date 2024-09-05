import torch
import torchvision
import matplotlib.pyplot as plt

import os
import yaml
import numpy as np

import sys
from lfads import LFADS_Net
from utils import read_data, load_parameters, save_parameters, batchify_random_sample

# Select device to train LFADS on
device = 'cuda' if torch.cuda.is_available() else 'cpu'; print('Using device: %s'%device)

window_len = 16
T_init = 4
num_unique_groups = 200
num_epoch = 3000
T_var = 8
import csv

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

################ load model #####################
def load_model(model_name_1):
    load_path_1 = 'models/optogenetic_simu_data_prediction_photostim/' + model_name_1
    hyperparams_1 = load_parameters(load_path_1+'parameters.yaml')
    save_parameters(hyperparams_1)

    T_init = 4
    model_1 = LFADS_Net(inputs_dim = num_cells, groups_dim = num_unique_groups, T = num_steps, T_init = T_init, dt = 0.05, device=device,
                     model_hyperparams=hyperparams_1).to(device)
    model_1.load_checkpoint('best', output = '')

    metrics_1 = {}
    T_eval_start = 4

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

# learning_type = 'random'
learning_type = 'active'
day = '0117'
analysis_day = '0201'
plot_loss_fig = True
save_results = False
test_group_size_list = ['mix']
group_size_list = ['mix']
# test_group_size_list = [1, 10, 20, 50, 100, 200, 'mix']
# group_size_list = [1, 10, 20, 50, 100, 200, 'mix']
# num_groups_list = [200, 250, 500, 1000, 2000, 5000, 10000]
num_groups_list = [200, 300]
target_untarget_metrics_all = {}
untarget_metrics_all = {}

for test_group_size in test_group_size_list:

    data = np.load(f'../data/{day}_test_random_selection_holdout_{test_group_size}_sim_data.npy', allow_pickle=True).item()

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

    if learning_type == 'random':
        for group_size in group_size_list:
            for num_groups in num_groups_list:
                fname = f'photostim_random_learning_g_1024_01_17_stim_group_{group_size}_num_groups_{num_groups}_rand_day_0/'
                print('------num groups:', num_groups)
                print('------test group size:', test_group_size)
                print('------group size:', group_size)
                model_1, metrics_1 = load_model(model_name_1 = fname)
                conditions  = f'group_size_{group_size}_test_group_size_{test_group_size}_num_groups_{num_groups}'
                print(conditions)
                target_untarget_metrics_all[conditions] = metrics_1['variance_explained_test']
                untarget_metrics_all[conditions] = metrics_1['variance_explained_test_untarget']

        if plot_loss_fig:
            fig, axs = plt.subplots(2, 2, figsize = (10, 8))
            for group_size in group_size_list:
                for num_groups in num_groups_list:
                    fname = f'photostim_random_learning_g_1024_01_17_stim_group_{group_size}_num_groups_{num_groups}_rand_day_0/'
                    load_path = 'models/optogenetic_simu_data_prediction_photostim/' + fname

                    epochs = []
                    train_losses = []
                    train_recon_losses = []
                    train_kld = []
                    train_l2 = []
                    valid_losses = []
                    valid_recon_losses = []
                    valid_kld = []
                    with open(load_path+'loss.csv', newline='') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            epochs.append(int(row['epoch']))
                            train_losses.append(float(row['train_loss']))
                            train_recon_losses.append(float(row['train_recon_loss']))
                            train_kld.append(float(row['train_kl_loss']))
                            train_l2.append(float(row['l2_loss']))
                            valid_losses.append(float(row['valid_loss']))
                            valid_recon_losses.append(float(row['valid_recon_loss']))
                            valid_kld.append(float(row['valid_kl_loss']))


                    # axs[0,0].plot(np.asarray(epochs), np.asarray(train_losses))
                    axs[0,0].plot(np.asarray(epochs), np.asarray(valid_losses))
                    axs[0,0].set_title('loss v.s. epoch')

                    # axs[0,1].plot(np.asarray(epochs), np.asarray(train_recon_losses))
                    axs[0,1].plot(np.asarray(epochs), np.asarray(valid_recon_losses))
                    axs[0,1].set_title('recon loss v.s. epoch')

                    # axs[1,0].plot(np.asarray(epochs), np.asarray(train_l2))
                    axs[1,0].plot(np.asarray(epochs), np.asarray(train_l2))
                    axs[1,0].set_title('l2 sparsity v.s. epoch')

                    # axs[1,1].plot(np.asarray(epochs), np.asarray(train_kld))
                    axs[1,1].plot(np.asarray(epochs), np.asarray(valid_kld))
                    axs[1,1].set_title('kld v.s. epoch')

            legend_list = []
            for group_size in group_size_list:
                for num_groups in num_groups_list: 
                    # legend_list.append(f'train-group-size-{group_size}')
                    legend_list.append(f'val-group-size-{group_size}-num_groups-{num_groups}')
            axs[0,0].legend(legend_list)
            plt.savefig(f'results/{analysis_day}/{analysis_day}_g_1024_loss_num_groups_varied_group_size_{group_size}_test_group_size_{test_group_size}_models.png')

        if save_results:
            np.save(f'results/{analysis_day}/{analysis_day}_{learning_type}_g_1024_target_untarget_metrics_all.npy', target_untarget_metrics_all)
            np.save(f'results/{analysis_day}/{analysis_day}_{learning_type}_g_1024_untarget_metrics_all.npy', untarget_metrics_all)
        
    else:
        sparse_list = [70.0]
        similarity_list = [100.0]
        num_target_neurons = 'mix'
        temp = 1.0
        input_type = 'continous'
        num_groups = 200
        num_active_groups_list = [300]
        
        for sparse in sparse_list:
            for similarity in similarity_list:
                for num_active_groups in num_active_groups_list:
                    fname = f"photostim_active_learning_g_1024_{analysis_day}_stim_group_{num_target_neurons}_num_groups_{num_groups}_num_active_groups_{num_active_groups}_sparse_{sparse}_similarity_{similarity}_temp_{temp}_input_type_{input_type}_day_1/"      
                    print(fname)
                    model_1, metrics_1 = load_model(model_name_1 = fname)
                    

                    if plot_loss_fig:
                        fig, axs = plt.subplots(2, 2, figsize = (10, 8))        
                        load_path = 'models/optogenetic_simu_data_prediction_photostim/' + fname

                        epochs = []
                        train_losses = []
                        train_recon_losses = []
                        train_kld = []
                        train_l2 = []
                        valid_losses = []
                        valid_recon_losses = []
                        valid_kld = []

                        with open(load_path+'loss.csv', newline='') as csvfile:
                            reader = csv.DictReader(csvfile)
                            for row in reader:
                                epochs.append(int(row['epoch']))
                                train_losses.append(float(row['train_loss']))
                                train_recon_losses.append(float(row['train_recon_loss']))
                                train_kld.append(float(row['train_kl_loss']))
                                train_l2.append(float(row['l2_loss']))
                                valid_losses.append(float(row['valid_loss']))
                                valid_recon_losses.append(float(row['valid_recon_loss']))
                                valid_kld.append(float(row['valid_kl_loss']))

                        axs[0,0].plot(np.asarray(epochs), np.asarray(valid_losses))
                        axs[0,0].set_title('loss v.s. epoch')

                        axs[0,1].plot(np.asarray(epochs), np.asarray(valid_recon_losses))
                        axs[0,1].set_title('recon loss v.s. epoch')

                        axs[1,0].plot(np.asarray(epochs), np.asarray(train_l2))
                        axs[1,0].set_title('l2 sparsity v.s. epoch')

                        axs[1,1].plot(np.asarray(epochs), np.asarray(valid_kld))
                        axs[1,1].set_title('kld v.s. epoch')
                        plt.savefig(f'results/{analysis_day}/{analysis_day}_g_1024_loss_active_learning_sparsity_{sparse}_test_group_size_{test_group_size}_models.png')