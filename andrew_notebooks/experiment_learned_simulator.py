import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import pickle
import input_design_cpu as input_design
import estimation_procedures_cpu as est
import os
import time
import sys

#torch.set_default_device('cuda:0')
torch.set_default_device('cpu')


def compute_transfer_matrix2(Ahat,rollout_len,k=5):
    d = Ahat.shape[0]
    avg_connect_ark = np.zeros((d,d))
    #rollout_len = 15
    params = []
    A_params = []
    B_params = []
    for i in range(k):
        params.append(np.zeros((d,d)))
        A_params.insert(0,Ahat[:,i*d:(i+1)*d])
        B_params.insert(0,Ahat[:,d*k+i*d:d*k+(i+1)*d])

    for t in range(rollout_len):
        param_new = np.zeros((d,d))
        for i in range(k):
            param_new += A_params[i] @ params[i]
        if t <= k-1:
            param_new += B_params[t] 
        params = params[:-1]
        params.insert(0,param_new)
        avg_connect_ark += params[0]
    return avg_connect_ark

def compute_transfer_gt(Ast_new, k):
    transfer_gt = compute_transfer_matrix2(Ast_new, rollout_len=15, k=k)
    transfer_gt += compute_transfer_matrix2(Ast_new, rollout_len=14, k=k)
    transfer_gt += compute_transfer_matrix2(Ast_new, rollout_len=13, k=k)
    return transfer_gt



def load_full_Ahat():
    Ast = np.load('./estimated_parameters/AR5_estimate.npy')
    num_neurons = NUM_NEURONS
    d = num_neurons
    k = 5
    Ast_new = np.zeros((num_neurons,2*k*num_neurons+1))
    for i in range(2*k):
        Ast_new[:,i*num_neurons:(i+1)*num_neurons] = Ast[0:num_neurons,i*d:i*d+num_neurons]
    Ast_new[:,-1] = Ast[0:num_neurons,-1]
    transfer_gt = compute_transfer_gt(Ast_new, k=k)
    return transfer_gt



def load_lowrank_Ahat(rank):
    #params = np.load('./estimated_parameters/Ahat_D_U_V_low_rank_' + str(rank) + '.npz.npy', allow_pickle=True)[()]
    params = np.load('./estimated_parameters/Ahat_D_U_V_low_rank_[' + rank + '].npz.npy', allow_pickle=True)[()]
    num_neurons = NUM_NEURONS
    d = num_neurons
    k = 4
    Ast_new = np.zeros((num_neurons,2*k*num_neurons+1))
    for i in range(2*k):
        if i < k:
            Ast_new[:,i*num_neurons:(i+1)*num_neurons] = np.diag(params['alpha'][i]) + params['W_u'][i] @ params['W_v'][i].T
        else:
            Ast_new[:,i*num_neurons:(i+1)*num_neurons] = np.diag(params['beta'][i-k]) + params['B_u'][i-k] @ params['B_v'][i-k].T
    transfer_gt = compute_transfer_gt(Ast_new, k=k)
    return transfer_gt





def compute_error(A_gt, A_est):
    A1 = A_gt - np.diag(np.diag(A_gt))
    A2 = A_est - np.diag(np.diag(A_est))
    e_no_diag = np.linalg.norm(A1 - A2, 'fro')
    return np.linalg.norm(A_gt - A_est, 'fro'), e_no_diag

def compute_pred_error(A_gt, A_est, max_on=10, no_diag=False):
    d = A_gt.shape[0]
    T = int(100*la.norm(A_gt, 2) + 100*la.norm(A_est, 2))
    mean_error = 0
    for t in range(T):
        u_idx = np.random.choice(range(d), size=max_on, replace=False)
        u = np.zeros(d)
        u[u_idx] = 1.0
        if no_diag:
            delta = (A_gt - A_est) @ u
            delta[u_idx] = 0
            mean_error += la.norm(delta, 2) / T
        else:
            mean_error += la.norm((A_gt - A_est) @ u, 2) / T
    return mean_error
        

def generate_rand_input(d,max_on):
    idx = np.random.choice(np.linspace(0,d-1,d).astype(int), size=max_on, replace=False)
    u = np.zeros(d)
    u[idx] = 1.0
    return u

def get_obs(A_gt, u, noise_std=0.25):
    d = A_gt.shape[0]
    return A_gt @ u + noise_std*np.random.randn(d)

def get_cov(inputs,start,cov_len):
    d = inputs.shape[0]
    cov = np.zeros((d,d))
    idx = start
    for i in range(cov_len):
        cov += np.outer(inputs[:,idx],inputs[:,idx])
        idx += 1
        if idx >= inputs.shape[1]:
            idx = 0
    return cov
    
def run_experiment_helper(params):
    data = {}
    data["time"] = []
    data["input_norm"] = 0
    for est_type in params["est_type"]:
        data[est_type + '_diag'] = []
        data[est_type + '_nodiag'] = []
        data[est_type + '_pred_error'] = []
    U = []
    Y = []
    d = params["d"]
    A_gt = params["A_gt"]
    A_nuc = None
    A_input = None
    input_cov = np.zeros((d,d))
    _,S_hat,V_hat = la.svd(A_gt)
    
    if params["type"] == "active" or params["type"] == "uniform":
        u_uniform, _ = input_design.design_inputs_constrained(np.random.randn(d,d),
                                                             n_iters=200,
                                                             l1_constraint=params["max_on"],
                                                             k=d,
                                                             num_batches=2250, #2000,
                                                             verbose=False,
                                                             V_design=True,
                                                             plt_save=params["plt_save"] + "_uniformdesign")
        u_uniform_cov = u_uniform @ u_uniform.T
        input_uniform_idx = 0
    
    if params["type"] == "oracle":
        V_design = (params["design_type"] == "V")
        input_idx = 0
        if params["no_diag"]:
            A_oracle_design = A_gt
        else:
            A_oracle_design = A_gt
        u_oracle, _ = input_design.design_inputs_constrained(A_oracle_design,
                                                         n_iters=100,
                                                         l1_constraint=params["max_on"],
                                                         k=params["inputs_r"],
                                                         num_batches=np.min([1000,params["T"]]),
                                                         verbose=False,
                                                         V_design=V_design,
                                                         plt_save=params["plt_save"] + "_inputdesign")
        
    for t in range(params["T"]):
        if params["type"] == "random":
            u = generate_rand_input(d, params["max_on"])
        elif params["type"] == "uniform":
            u = u_uniform[:,input_uniform_idx]
            input_uniform_idx += 1
            if input_uniform_idx >= u_uniform.shape[1]:
                input_uniform_idx = 0
        elif params["type"] == "oracle":
            if params["mix_rand"] and np.mod(t,2) == 0:
                u = generate_rand_input(d, params["max_on"])
            else:
                u = u_oracle[:,input_idx]
                input_idx += 1
                if input_idx >= u_oracle.shape[1]:
                    input_idx = 0
        elif params["type"] == "active":
            if (params["mix_rand"] and np.mod(t,4) == 0) or t <= params["active_update_interval"] + 1:
            #if (params["mix_rand"] and np.mod(t,4) == 0) or t <= params["record_interval"] + 1:
                #u = generate_rand_input(d, params["max_on"])
                u = u_uniform[:,input_uniform_idx]
                input_uniform_idx += 1
                if input_uniform_idx >= u_uniform.shape[1]:
                    input_uniform_idx = 0
            else:
                #input_idx = np.random.randint(0, high=1000)
                u = u_active[:,input_idx]
                input_idx += 1
                if input_idx >= u_active.shape[1]:
                    input_idx = 0
            if np.mod(t,params["active_update_interval"]) == 1 and t > 10:
            #if (np.mod(t, params["record_interval"]) == 1 and t > 10): # or t == 500:
                # A_input = est.estimate_ls(U,Y)
                #_,S_hat,V_hat = la.svd(A_input)
                #if A_input is None:
                #    A_input = est.estimate_ls(U,Y)
                #A_input = est.estimate_gd_nuc_project(U,Y,reg=params["nuc_reg"],lr=0.0001,n_iters=500,transfer0=A_input)
                print(t,'computing input')
                V_design = (params["design_type"] == "V")
                input_idx = 0
                if params["no_diag"]:
#                     A_ls = est.estimate_ls(U,Y)
#                     if t < 1000:
#                         n_iters = 1500
#                     else:
#                         n_iters = 500
#                     A_nuc, D_nuc, UV_nuc = est.estimate_gd_nuc_project_diag(U,Y,reg=150,lr=0.0001,n_iters=n_iters,transfer0=A_ls)
#                     A_input = UV_nuc
                    A_input = UV_nuc
                else:
                    #A_ls = est.estimate_ls(U,Y)
                    #A_nuc = est.estimate_gd_nuc_project(U,Y,reg=100,lr=0.0001,n_iters=500,transfer0=A_ls)
                    #A_lr = est.estimate_gd_lowrank_alternate2(U,Y,rank=params["rank_reg"],lr=0.01,n_iters=500,transfer0=A_ls)
                    A_input = A_nuc
#                 _,S_input,_ = la.svd(A_input)
#                 for sig_idx in range(len(S_input)):
#                     if np.sum(S_input[0:sig_idx]) / np.sum(S_input) > 0.9:
#                         inputs_r = sig_idx
#                         print(inputs_r)
#                         break
                #unif_cov = get_cov(u_uniform,input_uniform_idx,250)
#                 if t < 1000:
#                     num_batches = 375
#                 else:
                #V_true = input_design.est_V(U,Y,params["inputs_r"])
                num_batches = 750
                u_active, _ = input_design.design_inputs_constrained(A_input,
                                                                 n_iters=700,
                                                                 l1_constraint=params["max_on"],
                                                                 k=params["inputs_r"],
                                                                 num_batches=num_batches,
                                                                 verbose=False,
                                                                 V_design=V_design,
                                                                 #cov0=input_cov / num_batches,
                                                                 V_true=None,
                                                                 plt_save=params["plt_save"] + "_inputdesign_" + str(t))
        obs = get_obs(A_gt, u, noise_std=params["noise_std"])
        input_cov += np.outer(u,u)
        U.append(u)
        Y.append(obs)
        data["input_norm"] += np.linalg.norm(u, 1) / params["T"]
        
        if np.mod(t, params["record_interval"]) == 0 and t > 0:
            print("estimating, t = " + str(t))
            data["time"].append(t)
            A_ls = est.estimate_ls(U,Y)
            if "ls" in params["est_type"]:
                e1, e2 = compute_error(A_gt, A_ls)
                data["ls_diag"].append(e1)
                data["ls_nodiag"].append(e2)
                data["ls_pred_error"].append(compute_pred_error(A_gt, A_ls, no_diag=True))
            if "nuc" in params["est_type"]:
                print("estimating nuc")
                #nuc_iters = int(np.max([200, 1000*500/t + 100]))
                nuc_iters = int(1000*500/t)
                if t > 12000:
                    lr = 0.00002
                    nuc_iters = int(2*nuc_iters)
                elif t > 6000:
                    lr = 0.00005
                    nuc_iters = int(1.5*nuc_iters)
                else:
                    lr = 0.0001
                if params["no_diag"]:
                    A_nuc, D_nuc, UV_nuc = est.estimate_gd_nuc_project_diag(U,Y,reg=params["nuc_reg"],lr=lr,n_iters=nuc_iters,transfer0=A_ls,plt_save=params["plt_save"] + "_nuc_" + str(t))
                else:
                    A_nuc = est.estimate_gd_nuc_project(U,Y,reg=params["nuc_reg"],lr=0.0001,n_iters=nuc_iters,transfer0=A_ls,plt_save=params["plt_save"] + "_nuc_" + str(t))
                e1, e2 = compute_error(A_gt, A_nuc)
                data["nuc_diag"].append(e1)
                data["nuc_nodiag"].append(e2)
                data["nuc_pred_error"].append(compute_pred_error(A_gt, A_nuc, no_diag=True))
            if "nuc-ls" in params["est_type"]:
                if params["no_diag"]:
                    A_nuc_ls = refit_nuc_nodiag(UV_nuc,A_nuc,U,Y)
                else:
                    A_nuc_ls = refit_nuc(A_nuc,U,Y)
                e1, e2 = compute_error(A_gt, A_nuc_ls)
                data["nuc-ls_diag"].append(e1)
                data["nuc-ls_nodiag"].append(e2)
                data["nuc-ls_pred_error"].append(compute_pred_error(A_gt, A_nuc_ls, no_diag=True))
            if "lowrank" in params["est_type"]:
                print("estimating lowrank")
                if params["no_diag"]:
                    A_lr = est.estimate_gd_lowrank_alternate2_diag(U,Y,rank=params["rank_reg"],lr=0.01,n_iters=500,transfer0=A_ls)
                    #A_lr = est.estimate_gd_lowrank_project_diag(U,Y,rank=params["rank_reg"],lr=0.25,n_iters=2000,transfer0=A_ls)
                else:
                    A_lr = est.estimate_gd_lowrank_alternate2(U,Y,rank=params["rank_reg"],lr=0.01,n_iters=500,transfer0=A_ls)
                e1, e2 = compute_error(A_gt, A_lr)
                data["lowrank_diag"].append(e1) 
                data["lowrank_nodiag"].append(e2) 
                data["lowrank_pred_error"].append(compute_pred_error(A_gt, A_lr, no_diag=True))
    return data         




def run_experiment(params):
    no_diag = False
    # if params["gt_file"] == "d663_full":
    #     A_gt = load_full_Ahat()
    # elif params["gt_file"] == "d663_lr15":
    #     A_gt = load_lowrank_Ahat(params['gt_file'])
    #     no_diag = True
    # elif params["gt_file"] == "d663_lr35":
    #     A_gt = load_lowrank_Ahat(rank=35)
    #     no_diag = True
    if 'full' in params["gt_file"]:
        A_gt = load_full_Ahat(params['gt_file'])
    else:
        A_gt = load_lowrank_Ahat(params['gt_file'])
        no_diag = True
    print(np.linalg.norm(A_gt - np.diag(np.diag(A_gt)),'fro'))

    exp_params = {
        "A_gt": A_gt,
        "T": params["T"],
        "record_interval": params["record_interval"],
        "est_type": params["est_type"],
        "d": A_gt.shape[0],
        "max_on": params["max_on"],
        "mix_rand": True,
        "no_diag": no_diag,
        "design_type": "V",
        "noise_std": params["noise_std"],
        "active_update_interval": params["active_update_interval"]
    }

    if not os.path.isdir('./results/' + params["exp_id"]):
        os.makedirs('./results/' + params["exp_id"])
    if not os.path.isdir('./results/' + params["exp_id"] + '/plots/trial' + params['trial_idx']):
        os.makedirs('./results/' + params["exp_id"] + '/plots/trial' + params['trial_idx'])

    results = {}
    for i in range(params["num_trials"]):
        for exp_type in params["types"]:
            exp_params["type"] = exp_type
            for nuc_reg in params["nuc_reg"]:
                exp_params["nuc_reg"] = nuc_reg
                if exp_type == "random" or exp_type == "uniform":
                    name = exp_type + "-nr" + str(nuc_reg) + "-" + str(i)
                    exp_params["plt_save"] = './results/' + params["exp_id"] + '/plots/trial' + params['trial_idx'] + '/' + name
                    result_ = run_experiment_helper(exp_params)
                    results[name] = result_
                elif exp_type == "active" or exp_type == "oracle":
                    for input_r in params["inputs_r"]:
                        name = exp_type + "-nr" + str(nuc_reg) + "-inr" + str(input_r) + "-" + str(i)
                        exp_params["inputs_r"] = input_r
                        exp_params["plt_save"] = './results/' + params["exp_id"] + '/plots/trial' + params['trial_idx'] + '/' + name
                        result_ = run_experiment_helper(exp_params)
                        results[name] = result_
                with open('./results/' + params["exp_id"] + '/results_' + params["trial_idx"] + '.pkl', 'wb') as f:
                    pickle.dump([results,params], f)


'''
Supported file IDs:
     - d663_full: full rank, d=663
     - d663_lr15: rank 15, d=663
     - d663_lr35: rank 35, d=663
'''

seed = int(time.time())
np.random.seed(seed)
torch.manual_seed(seed)

trial_idx = sys.argv[1]


# 58 - 507
# 59 - 502
# 60 - 504
NUM_NEURONS = 663

params = {
    "exp_id": "test",
    "gt_file": "d663_r35",
    "T": 10001, # 5001
    "record_interval": 1000,
    "active_update_interval": 1000,
    "types": ["active","random","uniform"], # ["active","uniform","random","oracle"],
    "est_type": ["nuc","ls"],
    "max_on": 30, 
    "no_diag": False,
    "nuc_reg": [2.5,5,7.5,10],
    "inputs_r": [50,75,100], 
    "noise_std": 0.25, #0.4,
    "num_trials": 1, 
    "trial_idx": trial_idx,
    "seed": seed
}

run_experiment(params)

