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


class Dataset:
    def __init__(self, segments_y, segments_u, segments_nan, segments_spiking, train_idx, test_idx):
        self.segments_y = segments_y
        self.segments_u = segments_u
        self.segments_nan = segments_nan
        self.segments_spiking = segments_spiking
        self.train_indices = train_idx  
        self.test_indices = test_idx
        self.d = segments_y[0].shape[1]
        self.ark_order = 1
        self.roc_thresholds = np.linspace(-2,5,15)
        self.compute_test_quantities()

    def get_y(self,idx):
        return self.segments_y[idx].copy()

    def get_u(self,idx):
        return self.segments_u[idx].copy()

    def get_train_idx(self):
        return self.train_indices.copy()

    def get_inputs(self):
        cov_u = np.zeros((self.d,len(self.segments_u)))
        for i in range(len(self.segments_u)):
            for t in range(self.segments_u[i].shape[0]):
                if np.sum(self.segments_u[i][t,:]) > 0:
                    cov_u[:,i] = self.segments_u[i][t,:].copy()
                    break
        return cov_u.T

    def compute_test_quantities(self):
        x_true = []
        spike_label = []
        input_segment = []
        for i in range(len(self.test_indices)):
            if self.test_indices[i]:
                x_true.append([])
                spike_label.append([])
                input_segment.append(0)
                for t in range(self.segments_y[i].shape[0]):
                    if t >= self.ark_order:
                        if not self.segments_nan[i][t]:
                            x_true[-1].append(self.segments_y[i][t,:].copy().flatten())
                            spike_label[-1].append(self.segments_spiking[i][t,:].copy().flatten())
                input_segment[-1] = np.sum(self.segments_u[i], axis=0) > 0
                x_true[-1] = np.array(x_true[-1])
                spike_label[-1] = np.array(spike_label[-1])
        self.test_true = x_true
        self.test_spike_label = spike_label
        self.test_input_segment = input_segment


    def compute_roc(self, prediction):
        tpr = []
        fpr = []
        tpr_noinput = []
        fpr_noinput = []

        for thresholds_idx in range(len(self.roc_thresholds)):
            for mode_idx in range(2):
                tp_total = 0
                fp_total = 0
                p_total = 0
                n_total = 0

                for neuron in range(self.d):
                    output_pred = []
                    output_true = []
                    output_label = []
                    sequence_idx = 0
                    for i in range(len(self.test_true)):
                        if mode_idx == 0:
                            output_pred.extend(prediction[i][:,neuron])
                            output_true.extend(self.test_true[i][:,neuron])
                            output_label.extend(self.test_spike_label[i][:,neuron])
                        elif not self.test_input_segment[i][neuron]:
                            output_pred.extend(prediction[i][:,neuron])
                            output_true.extend(self.test_true[i][:,neuron])
                            output_label.extend(self.test_spike_label[i][:,neuron])
                    output_pred = np.array(output_pred)
                    output_true = np.array(output_true)
                    output_label = np.array(output_label).astype(bool)

                    mean_threshold = np.median(output_true)
                    lower_tail_idx = (output_true < mean_threshold)
                    lower_tail_data = output_true[lower_tail_idx]
                    lower_tail_std = np.std(lower_tail_data)
                    detect_spike_threshold = mean_threshold + self.roc_thresholds[thresholds_idx]*lower_tail_std

                    predicted_spikes = (output_pred > detect_spike_threshold)
                    tp_total += np.sum(np.logical_and(predicted_spikes,output_label))
                    fp_total += np.sum(np.logical_and(predicted_spikes,~output_label))
                    p_total += np.sum(output_label)
                    n_total += np.sum(~output_label)

                if mode_idx == 0:
                    tpr.append(tp_total / p_total)
                    fpr.append(fp_total / n_total)
                else:
                    if p_total > 0:
                        tpr_noinput.append(tp_total / p_total)
                    if n_total > 0:
                        fpr_noinput.append(fp_total / n_total)
        return tpr, fpr, tpr_noinput, fpr_noinput


    def evaluate(self, Ahat, roc=True):
        x_pred = []
        mse = 0
        mse_pc1 = 0
        count = 0
        for i in range(len(self.test_indices)):
            if self.test_indices[i]:
                x_past = []
                x_pred.append([])
                for t in range(self.segments_y[i].shape[0]):
                    if t < self.ark_order:
                        x_past.append(self.segments_y[i][t,:].copy().flatten())
                    else:
                        z = np.array(x_past).flatten()
                        z = np.concatenate((z,self.segments_u[i][t-self.ark_order:t,:].copy().flatten(),np.ones(1)))
                        x_next = Ahat @ z
                        x_past.pop(0)
                        x_past.append(x_next.copy())
                        if not self.segments_nan[i][t]:
                            x_pred[-1].append(x_next.copy())
                            mse += np.linalg.norm(x_next.copy() - self.segments_y[i][t,:].copy().flatten(), 2)**2
                            #mse_pc1 += (y_pc1.T @ (x_next.copy() - self.segments_y[i][t,:].copy().flatten()))**2
                            count += 1
                x_pred[-1] = np.array(x_pred[-1])
        if roc:
            tpr, fpr, tpr_noin, fpr_noin = self.compute_roc(x_pred)
            return mse / count, tpr, fpr, tpr_noin, fpr_noin
        else:
            return mse / count





def load_data(filename='sample_photostim_0113.npy'):
    # trial data
    data = np.load('./data/' + filename, allow_pickle = True).item()
    #data = np.load('./data/photostim_0404/photostim_0404_date_070623.npy', allow_pickle=True).item()
    #y = data['y']
    #u = data['u']
    # num_trials = y.shape[0]
    # num_steps = y.shape[1]
    # num_neurons = y.shape[2]
    # d = num_neurons
    print(data['photostim'].keys())
    y_session = data['y_session'][4:,:]
    u_session = data['u_session'][4:,:]
    num_neurons = u_session.shape[1]
    d = num_neurons
    print(d)

    y_ses_nan = np.isnan(y_session)
    y_ses_nan_sum = np.sum(y_ses_nan, axis=1)
    y_ses_nan_idx = (y_ses_nan_sum > 0)

    #linear interpolation of data
    y_session_interp = y_session.copy()
    for i in range(y_session.shape[1]):
        nan_start = -1
        nan_stop = -1
        for j in range(y_session.shape[0]):
            if nan_start == -1 and np.isnan(y_session_interp[j,i]):
                nan_start = j - 1
            if nan_start != -1 and not np.isnan(y_session_interp[j,i]):
                nan_stop = j
            if nan_start != -1 and nan_stop != -1:
                slope = y_session_interp[nan_stop,i] - y_session_interp[nan_start,i]
                for k in range(nan_stop - nan_start - 1):
                    y_session_interp[nan_start + k + 1,i] = slope*k/(nan_stop-nan_start-1) + y_session_interp[nan_start,i]
                nan_start = -1
                nan_stop = -1
    print('number nan = ' + str(np.sum(np.isnan(y_session_interp))))

    spiking = np.zeros(u_session.shape).astype(bool)
    for neuron in range(u_session.shape[1]):
        output_true = y_session_interp[4:,neuron]
        mean_threshold = np.median(output_true)
        lower_tail_idx = (output_true < mean_threshold)
        lower_tail_data = output_true[lower_tail_idx]
        lower_tail_std = np.std(lower_tail_data)
        true_spike_threshold = mean_threshold + 6*lower_tail_std
        true_spikes = (output_true > true_spike_threshold)
        spiking[4:,neuron] = true_spikes  

    return split_data(u_session, y_session_interp, y_ses_nan_idx, spiking)


def split_data(u_session, y_session_interp, y_nan_idx, spiking):
    segments_y = []
    segments_u = []
    segments_nan = []
    segments_spiking = []
    t = 0
    saved_t = 0
    saved_idx = np.zeros(u_session.shape[0])
    while True:
        #if np.sum(u_session[t,:]) > 0 and np.sum(u_session[t-1,:]) == 0 and t < saved_t + 5:
        t2 = 5
        while True:
            if np.sum(u_session[saved_t+t2,:]) > 0 and np.sum(u_session[saved_t+t2-1,:]) == 0:
                break
            t2 += 1
            if saved_t+t2 == u_session.shape[0] - 1:
                break
        segment_stop = np.min([saved_t+t2-3, u_session.shape[0]])
        seg_y = y_session_interp[saved_t:segment_stop,:].copy() / 1000
        seg_u = u_session[saved_t:segment_stop,:].copy()
        saved_idx[saved_t:segment_stop] += 1.0
        nan_idx = y_nan_idx[saved_t:segment_stop]
        segments_y.append(seg_y)
        segments_u.append(seg_u)
        segments_nan.append(nan_idx)
        segments_spiking.append(spiking[saved_t:segment_stop,:])
        saved_t = segment_stop
        if saved_t + 5 >= u_session.shape[0]:
            break
    print(saved_idx.sum(),u_session.shape[0],np.sum(saved_idx > 1),len(segments_u))

    patterns = []
    pattern_count = []
    pattern_idx = []
    pattern_length = []
    d = u_session.shape[1]

    for i in range(len(segments_u)):
        found_pat = False
        for t in range(segments_u[i].shape[0]):
            if np.sum(np.abs(segments_u[i][t,:])) > 0:
                idx = np.linspace(0,d-1,d).astype(int)
                on = segments_u[i][t,:] > 0
                pattern = np.array(idx[on])
                found = False
                for j in range(len(patterns)):
                    if len(pattern) == len(patterns[j]):
                        if np.linalg.norm(pattern - patterns[j]) == 0:
                            pattern_count[j] += 1
                            pattern_idx[j].append(i)
                            found = True
                            break
                if found is False:
                    patterns.append(pattern)
                    pattern_count.append(1)
                    pattern_idx.append([i])
                    pattern_length.append(len(pattern))
                found_pat = True
                break
        if not found_pat:
            print('error',i,segments_u[i].sum())

                
    # remove patterns randomly

    num_patterns = 20
    removed_patterns = []
    removed_count = 0
    train_idx = np.ones(len(segments_u)).astype(int)
    test_idx = np.zeros(len(segments_u)).astype(int)

    while len(removed_patterns) < num_patterns:
        p_idx = np.random.randint(0,len(patterns))
        if p_idx not in removed_patterns:
            removed_patterns.append(p_idx)
            removed_count += pattern_count[p_idx]
            test_idx[pattern_idx[p_idx]] = 1
            train_idx[pattern_idx[p_idx]] = 0

    print(removed_patterns)
            
    data = Dataset(segments_y, segments_u, segments_nan, segments_spiking, train_idx, test_idx)
    return data














def run_experiment_helper(params, data):
    results = {}
    results["time"] = []
    for est_type in params["est_type"]:
        results[est_type + '_mse'] = []
        #data[est_type + '_mse_pc1'] = []
        if params["roc"]:
            results[est_type + "_roc"] = []
            results[est_type + "_roc_noin"] = []
    X = []
    X2 = []
    Y = []
    d = data.d
    input_cov = 0.000001 * np.eye(d)
    state_cov = 0.000001 * np.eye(d)
    if params["type"] == "oracle":
        _,_,V_true = la.svd(params["B_gt"])
        V_true = V_true[0:params["inputs_r"],:]
    train_u = data.get_inputs()
    available_idx = data.get_train_idx()
    N = len(available_idx)

    t = -1
    while available_idx.sum() > 0:
        t += 1
        if params["type"] == "random":
            query_idx = np.random.choice(range(N), p=available_idx/available_idx.sum())
        elif params["type"] == "sequential":
            query_idx = t
        elif params["type"] == "oracle":
            if np.mod(t,2) > 0:
                input_cov_inv = la.inv(V_true @ input_cov @ V_true.T)
                cov_u = input_cov_inv @ V_true @ train_u.T
                ip = np.sum(cov_u * cov_u, axis=0)
                ip = np.multiply(ip,available_idx)
                ip[np.isnan(ip)] = 0
                query_idx = np.argmax(ip)
            else:
                query_idx = np.random.choice(range(N), p=available_idx/available_idx.sum())
        elif params["type"] == "active":
            if np.mod(t,2) > 0 and t > params["active_start"]:
                input_cov_inv = la.inv(V_hat @ input_cov @ V_hat.T)
                cov_u = input_cov_inv @ V_hat @ train_u.T
                ip = np.sum(cov_u * cov_u, axis=0)
                ip = np.multiply(ip,available_idx)
                ip[np.isnan(ip)] = 0
                query_idx = np.argmax(ip)
            else:
                query_idx = np.random.choice(range(N), p=available_idx/available_idx.sum())
        
        new_u = data.get_u(query_idx)
        new_y = data.get_y(query_idx)
        for i in range(new_u.shape[0] - 1):
            X.append(np.concatenate((new_u[i,:].flatten(), new_y[i,:].flatten())))
            X2.append(np.concatenate((new_u[i,:].flatten(), new_y[i,:].flatten(), np.ones(1))))
            Y.append(new_y[i+1,:].flatten())
            #state_cov += np.outer(train_segments_y[query_idx][i,:].copy().flatten(), train_segments_y[query_idx][i,:].copy().flatten())
        input_cov += np.outer(train_u[query_idx,:], train_u[query_idx,:])
        available_idx[query_idx] = 0
        
        if (np.mod(t, params["record_interval"]) == 0 and t > 0) or np.sum(available_idx) == 0:
            print('estimating', t)
            Xnp = np.array(X)
            X2np = np.array(X2)
            Ynp = np.array(Y)
            A_ls = np.linalg.pinv(X2np.T @ X2np + 0.00001 * np.eye(X2np.shape[1])) @ X2np.T @ Ynp
            A_ls = A_ls.T
            A_ls_eval = A_ls.copy()
            A_ls_eval[:,0:d] = A_ls_eval[:,d:2*d]
            A_ls_eval[:,d:2*d] = A_ls[:,0:d]
            results["time"].append(t)
            if params["type"] == "active":
                _,_,V_hat = la.svd(A_ls[:,0:d])
                V_hat = V_hat[0:params["inputs_r"],:]
            
            if "ls" in params["est_type"]:
                if params["roc"]:
                    mse, tpr, fpr, tpr_noin, fpr_noin = data.evaluate(A_ls_eval, roc=True)
                    results["ls_roc"].append([fpr,tpr])
                    results["ls_roc_noin"].append([fpr_noin,tpr_noin])
                else:
                    mse = data.evaluate(A_ls_eval, roc=False)
                results["ls_mse"].append(mse)
                #data["ls_mse_pc1"].append(mse_pc1)
            if "nuc" in params["est_type"]:
                A_nuc,B_nuc,v_nuc = est.estimate_gd_nuc_project_AB_offset(Xnp,Ynp,reg=params["nuc_reg"],transfer0=[A_ls[:,d:2*d],A_ls[:,0:d],A_ls[:,-1]],n_iters=1500,lr=0.1,plt_save=params['plt_save'] + "_nuc_" + str(t))
                G_nuc = np.concatenate((A_nuc,B_nuc,v_nuc[:,None]), axis=1)
                if params["roc"]:
                    mse, tpr, fpr, tpr_noin, fpr_noin = data.evaluate(G_nuc, roc=True)
                    results["nuc_roc"].append([fpr,tpr])
                    results["nuc_roc_noin"].append([fpr_noin,tpr_noin])
                else:
                    mse = data.evaluate(G_nuc, roc=False)
                results["nuc_mse"].append(mse)
                #data["nuc_mse_pc1"].append(mse_pc1)
    return results





def add_to_results(results, new_result, name, filename, params):
    if name in results.keys():
        results[name].append(new_result)
    else:
        results[name] = [new_result]
    with open(filename, 'wb') as f:
        pickle.dump([results,params], f)
    return results




def run_experiment(params, data):
    exp_params = {
        "record_interval": params["record_interval"],
        "est_type": params["est_type"],
        "roc": params["roc"]
    }

    if not os.path.isdir('./results_real/' + params["exp_id"]):
        os.makedirs('./results_real/' + params["exp_id"])
    if not os.path.isdir('./results_real/' + params["exp_id"] + '/plots/trial' + params['trial_idx']):
        os.makedirs('./results_real/' + params["exp_id"] + '/plots/trial' + params['trial_idx'])
    filename = './results_real/' + params["exp_id"] + '/results_' + params["trial_idx"] + '.pkl'

    results = {}
    for i in range(params["num_trials"]):
        for exp_type in params["types"]:
            exp_params["type"] = exp_type
            for nuc_reg in params["nuc_reg"]:
                exp_params["nuc_reg"] = nuc_reg
                if exp_type == "random" or exp_type == "uniform":
                    name = exp_type + "-nr" + str(nuc_reg)
                    print(name)
                    exp_params["plt_save"] = './results_real/' + params["exp_id"] + '/plots/trial' + params['trial_idx'] + '/' + name
                    result_ = run_experiment_helper(exp_params, data)
                    #results[name] = result_
                    results = add_to_results(results, result_, name, filename, params)
                elif exp_type == "active" or exp_type == "oracle":
                    for input_r in params["inputs_r"]:
                        for active_start in params["active_start"]:
                            name = exp_type + "-nr" + str(nuc_reg) + "-inr" + str(input_r) + "-astart" + str(active_start)
                            print(name)
                            exp_params["inputs_r"] = input_r
                            exp_params["active_start"] = active_start
                            exp_params["plt_save"] = './results_real/' + params["exp_id"] + '/plots/trial' + params['trial_idx'] + '/' + name
                            result_ = run_experiment_helper(exp_params, data)
                            #results[name] = result_
                            results = add_to_results(results, result_, name, filename, params)




#pattern_seed = 25 #int(time.time())


pattern_seed = int(sys.argv[1])
trial_idx = sys.argv[2]
np.random.seed(pattern_seed)
torch.manual_seed(pattern_seed)

data_file = 'BCI79_042324.npy'
#data_file = 'photostim_0404/photostim_0404_date_071323.npy'
#data_file = 'sample_photostim_58_spatial_date_071223.npy'
data = load_data(data_file)

seed = int(time.time())
np.random.seed(seed)
torch.manual_seed(seed)

params = {
    "exp_id": "BCI79_" + str(pattern_seed) + "_sweepr",
    "types": ["active","random"],
    "num_trials": 20,
    "record_interval": 200,
    "est_type": ["ls"],
    "roc": False,
    "nuc_reg": [None],
    "inputs_r": [25, 50, 75, 100, 125, 150], 
    "active_start": [200,400],
    "seed": seed,
    "trial_idx": trial_idx,
    "patterns_seed": pattern_seed,
}


run_experiment(params, data)

