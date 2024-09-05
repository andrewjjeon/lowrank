import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import pickle
import input_design as input_design
import estimation_procedures as est
import os
import time
import sys

torch.set_default_device('cuda:0')
#torch.set_default_device('cpu')

def np_sigmoid(z):
    return 1/(1 + np.exp(-z))


# def compute_roc(What, env):
#     Wtrue = env.W

#     thresholds = np.linspace(-1,5,20)
#     tprs = []
#     fprs = []

#     for t in range(len(thresholds)):
#         Wtrue_std = np.std(Wtrue.flatten())
#         Wtrue_threshold = np.logical_or(Wtrue > Wtrue_std + Wtrue.mean(), Wtrue < -Wtrue_std + Wtrue.mean())

#         What2 = What - np.diag(np.diag(What))
#         What_std = np.std(What2.flatten())
#         tolerance = thresholds[t]
#         What_threshold = np.logical_or(What2 > tolerance*What_std + What2.mean(), What < -tolerance*What_std + What2.mean())

#         tpr = np.sum(np.logical_and(Wtrue_threshold, What_threshold)) / np.sum(Wtrue_threshold)
#         fpr = np.sum(np.logical_and(~Wtrue_threshold, What_threshold)) / np.sum(~Wtrue_threshold)
#         tprs.append(tpr)
#         fprs.append(fpr)
#     return np.array(fprs), np.array(tprs)
def compute_transfer_gt(Ast_new, k=1):
    transfer_gt = compute_transfer_matrix2(Ast_new, rollout_len=15, k=k)
    transfer_gt += compute_transfer_matrix2(Ast_new, rollout_len=14, k=k)
    transfer_gt += compute_transfer_matrix2(Ast_new, rollout_len=13, k=k)
    return transfer_gt

def compute_roc(Ghat, env, compute_transfer=True):
    Wtrue = env.W
    if compute_transfer:
        num_neurons = Ghat.shape[0]
        d = num_neurons
        k = 1
        Ast_new = np.zeros((num_neurons,2*k*num_neurons+1))
        for i in range(2*k):
            Ast_new[:,i*num_neurons:(i+1)*num_neurons] = Ghat[0:num_neurons,i*d:i*d+num_neurons]
        Ast_new[:,-1] = Ghat[0:num_neurons,-1]
        What = compute_transfer_gt(Ast_new)
    else:
        What = Ghat
    
    thresholds = np.linspace(-1,5,20)
    tprs = []
    fprs = []

    for t in range(len(thresholds)):
        Wtrue_std = np.std(Wtrue.flatten())
        Wtrue_threshold = np.logical_or(Wtrue > Wtrue_std + Wtrue.mean(), Wtrue < -Wtrue_std + Wtrue.mean())

        What2 = What - np.diag(np.diag(What))
        What_std = np.std(What2.flatten())
        tolerance = thresholds[t]
        What_threshold = np.logical_or(What2 > tolerance*What_std + What2.mean(), What < -tolerance*What_std + What2.mean())

        tpr = np.sum(np.logical_and(Wtrue_threshold, What_threshold)) / np.sum(Wtrue_threshold)
        fpr = np.sum(np.logical_and(~Wtrue_threshold, What_threshold)) / np.sum(~Wtrue_threshold)
        tprs.append(tpr)
        fprs.append(fpr)
    return np.array(fprs), np.array(tprs)


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

# def compute_transfer_gt(Ast_new, k):
#     transfer_gt = compute_transfer_matrix2(Ast_new, rollout_len=15, k=k)
#     return transfer_gt



def compute_error(A_gt, A_est):
    A1 = A_gt - np.diag(np.diag(A_gt))
    A2 = A_est - np.diag(np.diag(A_est))
    e_no_diag = np.linalg.norm(A1 - A2, 'fro')
    return np.linalg.norm(A_gt - A_est, 'fro'), e_no_diag

def compute_pred_error(Ahat, snn, nonlinear=True):
    y_eval, x_eval, u_eval = snn.get_eval()
    mse = 0
    T = y_eval.shape[0]
    if nonlinear:
        for t in range(T):
            x_pred = Ahat[3] * np_sigmoid(Ahat[0] @ x_eval[t,:] + Ahat[1] @ u_eval[t,:] + Ahat[2])
            mse += np.linalg.norm(x_pred - y_eval[t,:], 2)**2
    else:
        for t in range(T):
            x_pred = Ahat[0] @ x_eval[t,:] + Ahat[1] @ u_eval[t,:] + Ahat[2]
            mse += np.linalg.norm(x_pred - y_eval[t,:], 2)**2
    return mse / np.linalg.norm(y_eval, 'fro')**2
        

def generate_rand_input(d,max_on):
    idx = np.random.choice(np.linspace(0,d-1,d).astype(int), size=max_on, replace=False)
    u = np.zeros(d)
    u[idx] = 1.0
    return u

def get_obs(snn, u):
    #x1, x0, _ = snn.get_snn_observation(u)
    #return x1, x0
    x, u = snn.get_snn_observation(200*u, onestep=False)
    return x, u

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

def clip_inputs(U,max_on):
    for i in range(U.shape[1]):
        idx = np.argsort(U[:,i])
        u_new = np.zeros(U.shape[0])
        for j in range(max_on):
            u_new[idx[-(j+1)]] = 1.0
        U[:,i] = u_new
    return U
    
def run_experiment_helper(params, env):
    data = {}
    data["time"] = []
    data["input_norm"] = 0
    for est_type in params["est_type"]:
        data[est_type + '_diag'] = []
        data[est_type + '_nodiag'] = []
        data[est_type + '_pred_error'] = []
        data[est_type + '_roc'] = []
    X = []
    X2 = []
    Y = []
    d = env.W.shape[0]
    A_nuc = None
    A_input = None
    input_cov = np.zeros((d,d))
    
    if params["type"] == "active" or params["type"] == "uniform":
        u_uniform, _ = input_design.design_inputs_constrained(np.random.randn(d,d),
                                                             n_iters=50,
                                                             l1_constraint=params["max_on"],
                                                             k=d,
                                                             num_batches=2250, #2000,
                                                             verbose=False,
                                                             V_design=True,
                                                             plt_save=params["plt_save"] + "_uniformdesign")
        u_uniform_cov = u_uniform @ u_uniform.T
        input_uniform_idx = 0
    
    if params["type"] == "oracle":
        input_idx = 0
        A_gt = params["A_gt"]
        if params["no_diag"]:
            A_oracle_design = A_gt
        else:
            A_oracle_design = A_gt
        u_oracle, _ = input_design.design_inputs_constrained(A_oracle_design,
                                                         n_iters=25,
                                                         l1_constraint=params["max_on"],
                                                         k=params["inputs_r"],
                                                         num_batches=np.min([1000,int(params["T"]/2)]),
                                                         verbose=False,
                                                         V_design=True,
                                                         plt_save=params["plt_save"] + "_inputdesign")
        u_oracle = clip_inputs(u_oracle, params["max_on"])
        
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
                print(t,'computing input')
                V_design = (params["design_type"] == "V")
                input_idx = 0
                if params["no_diag"]:
                    A_input = UV_nuc
                else:
                    A_input = A_nuc
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
        input_cov += np.outer(u,u)
        # y, x = get_obs(env, u)
        # X.append(np.concatenate((u,x)))
        # X2.append(np.concatenate((u,x,np.ones(1))))
        # Y.append(y)
        yt, ut = get_obs(env, u)
        for i in range(yt.shape[1]-1):
            X.append(np.concatenate((ut[:,i].flatten(),yt[:,i].flatten())))
            X2.append(np.concatenate((ut[:,i].flatten(),yt[:,i].flatten(),np.ones(1))))
            Y.append(yt[:,i+1].flatten())
        data["input_norm"] += np.linalg.norm(u, 1) / params["T"]
        
        if np.mod(t, params["record_interval"]) == 0 and t > 0:
            print("estimating, t = " + str(t))
            data["time"].append(t)
            Xnp = np.array(X)
            X2np = np.array(X2)
            Ynp = np.array(Y)
            if "ls" in params["est_type"]:
                A_ls = np.linalg.pinv(X2np.T @ X2np + 0.00001 * np.eye(X2np.shape[1])) @ X2np.T @ Ynp
                A_ls = A_ls.T
                B_ls = A_ls[:,0:d]
                A_ls_eval = A_ls.copy()
                A_ls_eval[:,0:d] = A_ls_eval[:,d:2*d]
                A_ls_eval[:,d:2*d] = A_ls[:,0:d]
                A_ls = A_ls[:,d:2*d]
                v_ls = A_ls[:,-1]
                #A_ls,B_ls,v_ls = est.estimate_gd_nuc_project_AB_offset(Xnp,Ynp,n_iters=2000,lr=0.1,plt_save=params['plt_save'] + "_nuc_" + str(t))
                #A_ls,B_ls,v_ls,scale_ls = est.estimate_gd_nuc_project_AB_offset_sigmoid(Xnp,Ynp,n_iters=2000,lr=0.1,plt_save=params['plt_save'] + "_nuc_" + str(t))
                
                # data["ls_diag"].append(e1)
                # data["ls_nodiag"].append(e2)
                #data["ls_pred_error"].append(compute_pred_error([A_ls,B_ls,v_ls], env, nonlinear=False))
                #data["ls_pred_error"].append(compute_pred_error([A_ls,B_ls,v_ls,scale_ls], env))
                fpr, tpr = compute_roc(B_ls, env, compute_transfer=False)
                data["ls_roc"].append([fpr, tpr])
            if "nl" in params["est_type"]:
                #A_ls,B_ls,v_ls = est.estimate_gd_nuc_project_AB_offset(Xnp,Ynp,n_iters=2000,lr=0.1,plt_save=params['plt_save'] + "_nuc_" + str(t))
                A_ls,B_ls,v_ls,scale_ls = est.estimate_gd_nuc_project_AB_offset_sigmoid(Xnp,Ynp,n_iters=2000,lr=0.1,plt_save=params['plt_save'] + "_nuc_" + str(t))
                
                # data["ls_diag"].append(e1)
                # data["ls_nodiag"].append(e2)
                #data["ls_pred_error"].append(compute_pred_error([A_ls,B_ls,v_ls], env, nonlinear=False))
                #data["ls_pred_error"].append(compute_pred_error([A_ls,B_ls,v_ls,scale_ls], env))
                A_nl_eval = np.zeros((d,2*d+1))
                A_nl_eval[:,0:d] = A_ls
                A_nl_eval[:,d:2*d] = B_ls
                A_nl_eval[:,-1] = v_ls
                fpr, tpr = compute_roc(A_nl_eval, env, compute_transfer=True)
                data["nl_roc"].append([fpr, tpr])
            if "nl_nuc" in params["est_type"]:
                print("estimating nuc")
                if "ls" in params["est_type"]:
                    A_nuc,B_nuc,v_nuc,scale_nuc = est.estimate_gd_nuc_project_AB_offset_sigmoid(Xnp,Ynp,transfer0=[A_ls,B_ls,v_ls,scale_ls],nuc_reg=True,reg=params["nuc_reg"],n_iters=1000,lr=0.1,plt_save=params['plt_save'] + "_nuc_" + str(t))
                else:
                    A_nuc,B_nuc,v_nuc,scale_nuc = est.estimate_gd_nuc_project_AB_offset_sigmoid(Xnp,Ynp,nuc_reg=True,reg=params["nuc_reg"],n_iters=1000,lr=0.1,plt_save=params['plt_save'] + "_nuc_" + str(t))
                #e1, e2 = compute_error(A_gt, A_nuc)
                #data["nuc_diag"].append(e1)
                #data["nuc_nodiag"].append(e2)
                #data["nl_nuc_pred_error"].append(compute_pred_error([A_nuc,B_nuc,v_nuc,scale_nuc], env))
                fpr, tpr = compute_roc(B_nuc, env)
                data["nl_nuc_roc"].append([fpr, tpr])
            if "nuc" in params["est_type"]:
                _,S,_ = la.svd(A_ls_eval[:,d:2*d])
                print(S.sum())
                A_nuc,B_nuc,v_nuc = est.estimate_gd_nuc_project_AB_offset(Xnp,Ynp,reg=params["nuc_reg"],transfer0=[A_ls_eval[:,0:d],A_ls_eval[:,d:2*d],A_ls[:,-1]],n_iters=2500,lr=0.0001,plt_save=params['plt_save'] + "_nuc_" + str(t))
                #A_nuc,B_nuc,v_nuc = est.estimate_gd_nuc_project_AB_offset(Xnp,Ynp,reg=params["nuc_reg"],transfer0=[A_ls[:,d:2*d],A_ls[:,0:d],A_ls[:,-1]],n_iters=1500,lr=0.01)
                G_nuc = np.concatenate((A_nuc,B_nuc,v_nuc[:,None]), axis=1)
                fpr, tpr = compute_roc(G_nuc, env, compute_transfer=True)
                data["nuc_roc"].append([fpr, tpr])
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

# seed = int(time.time())
# np.random.seed(seed)
# torch.manual_seed(seed)

# trial_idx = sys.argv[1]

# params = {
#     "exp_id": "photostim_0404_date_070623_r15_std0.5_2",
#     "gt_file": "photostim_0404_date_070623_r15",
#     "T": 10001, # 5001
#     "record_interval": 1000,
#     "active_update_interval": 1000,
#     "types": ["active","random","oracle","uniform"], # ["active","uniform","random","oracle"],
#     "est_type": ["nuc","ls"],
#     "max_on": 30, 
#     "no_diag": False,
#     "nuc_reg": [25,50,75,100], #[75,100,125,150],
#     "inputs_r": [15,25,50,75], #[10,25,50,75], 
#     "noise_std": 0.5, #0.4,
#     "num_trials": 1, 
#     "trial_idx": trial_idx,
#     "seed": seed
# }

# run_experiment(params)

