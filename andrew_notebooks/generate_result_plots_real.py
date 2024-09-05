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
import copy
import matplotlib

'''
best splits
58 - best: 210; worst: 16
59 - best: 308; worst: 0
60 - best: 411; worst: 2
663 - best: 17; worst: 6
'''

data_name = 'data60'
plot_name = '60'


if data_name == 'data0404':
    best_type = [
        'active-nrNone-inr125-astart200',
        'active-nrNone-inr150-astart200',
        'active-nrNone-inr150-astart200',
        'active-nrNone-inr125-astart200',
        'active-nrNone-inr125-astart400',
        'active-nrNone-inr125-astart200',
        'active-nrNone-inr100-astart200',
        'active-nrNone-inr125-astart200',
        'active-nrNone-inr75-astart400',
        'active-nrNone-inr150-astart400',
        'active-nrNone-inr150-astart400',
        None,
        'active-nrNone-inr125-astart200',
        'active-nrNone-inr125-astart400',
        'active-nrNone-inr75-astart400',
        'active-nrNone-inr125-astart200',
        'active-nrNone-inr150-astart200',
        'active-nrNone-inr100-astart400',
        'active-nrNone-inr125-astart400',
        'active-nrNone-inr150-astart200']
    omit = [11]
    string1 = 'data0404_071323_real_patterns10'
    string2 = 'data0404_071323_real_patterns1'
elif data_name == 'data663':
    best_type = [
        'active-nrNone-inr100-astart400',
        'active-nrNone-inr150-astart400',
        'active-nrNone-inr150-astart400',
        'active-nrNone-inr100-astart200',
        'active-nrNone-inr150-astart200',
        'active-nrNone-inr125-astart400',
        'active-nrNone-inr150-astart200',
        'active-nrNone-inr125-astart400',
        'active-nrNone-inr75-astart200',
        'active-nrNone-inr100-astart200',
        'active-nrNone-inr150-astart400',
        'active-nrNone-inr100-astart200',
        'active-nrNone-inr150-astart200',
        'active-nrNone-inr125-astart200',
        'active-nrNone-inr150-astart400',
        'active-nrNone-inr150-astart200',
        'active-nrNone-inr100-astart200',
        'active-nrNone-inr125-astart200',
        'active-nrNone-inr125-astart200',
        'active-nrNone-inr125-astart400'
    ]
    omit = []
    string1 = 'd663_real_patterns'
    string2 = string1
elif data_name == 'data58':
    omit = [3]
    best_type = [
        'active-nrNone-inr150-astart200',
        'active-nrNone-inr125-astart400',
        'active-nrNone-inr125-astart200',
        None,
        'active-nrNone-inr150-astart400',
        'active-nrNone-inr150-astart400',
        'active-nrNone-inr125-astart400',
        'active-nrNone-inr125-astart400',
        'active-nrNone-inr75-astart200',
        'active-nrNone-inr125-astart400',
        'active-nrNone-inr150-astart200',
        'active-nrNone-inr150-astart400',
        'active-nrNone-inr125-astart200',
        'active-nrNone-inr125-astart400',
        'active-nrNone-inr150-astart200',
        'active-nrNone-inr150-astart200',
        'active-nrNone-inr25-astart400',
        'active-nrNone-inr150-astart400',
        'active-nrNone-inr125-astart200',
        'active-nrNone-inr75-astart400',
        'active-nrNone-inr125-astart400']
    string1 = 'data58_real_patterns20'
    string2 = 'data58_real_patterns2'
elif data_name == 'data59':
    omit = []
    best_type = [
        'active-nrNone-inr25-astart400',
        'active-nrNone-inr100-astart200',
        'active-nrNone-inr125-astart200',
        'active-nrNone-inr150-astart200',
        'active-nrNone-inr150-astart200',
        'active-nrNone-inr150-astart200',
        'active-nrNone-inr50-astart200',
        'active-nrNone-inr25-astart200',
        'active-nrNone-inr25-astart200',
        'active-nrNone-inr125-astart200',
        'active-nrNone-inr150-astart200',
        'active-nrNone-inr50-astart200',
        'active-nrNone-inr50-astart200',
        'active-nrNone-inr75-astart200',
        'active-nrNone-inr75-astart200',
        'active-nrNone-inr50-astart200',
        'active-nrNone-inr50-astart200',
        'active-nrNone-inr50-astart200',
        'active-nrNone-inr75-astart400',
        'active-nrNone-inr75-astart200']
    string1 = 'data59_real_patterns30'
    string2 = 'data59_real_patterns3'
elif data_name == 'data60':
    omit = []
    best_type = [
        'active-nrNone-inr150-astart200',
        'active-nrNone-inr100-astart200',
        'active-nrNone-inr150-astart400',
        'active-nrNone-inr100-astart200',
        'active-nrNone-inr125-astart200',
        'active-nrNone-inr100-astart200',
        'active-nrNone-inr150-astart200',
        'active-nrNone-inr150-astart200',
        'active-nrNone-inr150-astart400',
        'active-nrNone-inr150-astart200',
        'active-nrNone-inr150-astart200',
        'active-nrNone-inr150-astart400',
        'active-nrNone-inr125-astart200',
        'active-nrNone-inr100-astart200',
        'active-nrNone-inr150-astart400',
        'active-nrNone-inr125-astart200',
        'active-nrNone-inr125-astart200',
        'active-nrNone-inr100-astart200',
        'active-nrNone-inr150-astart200',
        'active-nrNone-inr150-astart200']
    string1 = 'data60_real_patterns40'
    string2 = 'data60_real_patterns4'




results_best = {}
results_best_all = {}
results_best_all['active'] = []
results_best_all['random'] = []
results_best_count = {}
std_all_active = []
std_all_random = []
count = 0
time = None
roc_idx = 5

#types = ['active-nrNone-inr10','active-nrNone-inr25','active-nrNone-inr50','active-nrNone-inr75','active-nrNone-inr100','random-nrNone']
if True:
    j = 7
for file_idx in range(21):
    results = {}
    result_count = {}
#i = 12
    #if True:\
    if file_idx not in omit:
        if file_idx > 9:
            name = string2 + str(file_idx) + '_sweepr'
        else:
            name = string1 + str(file_idx) + '_sweepr'
        #name = 'd663_real_patterns' + str(j) + '_sweepr'
        files = os.listdir('./results_real/' + name)
        print(name)
        n = len(files)
        tmp = []

        file_results_passive = []
        file_results_active = []

        for i in range(len(files)):
            if 'results' in files[i]:
                with open('./results_real/' + name + '/' + files[i], 'rb') as f:
                    x = pickle.load(f)
                    count += len(x)
                    print(x[1])
                    for k in x[0].keys():
                        found_type = True
                        # found_type = False
                        # k_type = None
                        # for t in types:
                        #     if t in k:
                        #         found_type = True
                        #         k_type = t
                        # print(x[0][k].keys())
                        # if 'inr100' in k:
                        #     found_type = True
                        # else:
                        #     found_type = False
                        stop_idx = 0
                        if 'd663' in name and '_0' in files[i]:
                            stop_idx = 5
                        elif 'd663' in name and '_2' in files[i]:
                            stop_idx = 1
                        else:
                            stop_idx = len(x[0][k])
                            #print(len(x[0][k]))
                        for j in range(stop_idx):
                            for k2 in x[0][k][j].keys():
                                if 'ls_mse' in k2 and found_type:
                                    name2 = k + '_' + k2
                                    if k in results.keys():
                                        results[k] += np.array(x[0][k][j][k2])
                                        result_count[k] += 1
                                    else:
                                        results[k] = np.array(x[0][k][j][k2])
                                        result_count[k] = 1
                                if 'roc' in k2 and found_type:
                                    if 'roc_noin' in k2:
                                        name2 = k_type + '_roc_noin'
                                    else:
                                        name2 = k_type + '_roc'
                                    if name2 in results.keys():
                                        results[name2][0] += np.array(x[0][k][j][k2][roc_idx][0])
                                        results[name2][1] += np.array(x[0][k][j][k2][roc_idx][1])
                                        result_count[name2] += 1
                                    else:
                                        results[name2] = []
                                        results[name2].append(np.array(x[0][k][j][k2][roc_idx][0]))
                                        results[name2].append(np.array(x[0][k][j][k2][roc_idx][1]))
                                        result_count[name2] = 1
                                if 'ls_mse' in k2 and k in best_type[file_idx]:
                                    if 'active' in results_best.keys():
                                        results_best['active'] += np.array(x[0][k][j][k2])
                                        results_best_count['active'] += 1 
                                    else:
                                        results_best['active'] = np.array(x[0][k][j][k2])
                                        results_best_count['active'] = 1
                                    results_best_all['active'].append(np.array(x[0][k][j][k2]))
                                    file_results_active.append(np.array(x[0][k][j][k2]))
                                elif 'ls_mse' in k2 and 'random' in k:
                                    if 'random' in results_best.keys():
                                        results_best['random'] += np.array(x[0][k][j][k2])
                                        results_best_count['random'] += 1
                                    else:
                                        results_best['random'] = np.array(x[0][k][j][k2])
                                        results_best_count['random'] = 1
                                    results_best_all['random'].append(np.array(x[0][k][j][k2]))
                                    file_results_passive.append(np.array(x[0][k][j][k2]))

                            # if k == 'active-nr5-inr50-0':
                            #     tmp.append(x[0][k]['ls_mse'][4])
                            # plt.plot(x[0]['active-nrNone-inr50-0']['ls_roc'][5][0],x[0]['active-nrNone-inr50-0']['ls_roc'][5][1])
                            # plt.savefig('test.png')
                            # plt.close()
                            # quit()
                        time = x[0][k][0]['time']
        file_results_passive = np.array(file_results_passive)
        file_results_active = np.array(file_results_active)
        std_all_active.append(np.var(file_results_active, axis=0))
        std_all_random.append(np.var(file_results_passive, axis=0))
        if not os.path.isdir('./results_real/' + name + '/result_plots'):
            os.makedirs('./results_real/' + name + '/result_plots')


        # print(results.keys())
        # print(result_count)



        types = ['random','active']
        for t in types:
            count = 0
            plt.figure()
            for k in results.keys():
                if t in k:
                    if count > 9:
                        plt.plot(time[1:-1],results[k][1:-1] / result_count[k],'--',label=k)
                    else:
                        plt.plot(time[1:-1],results[k][1:-1] / result_count[k],label=k)
                    count += 1
            plt.legend()
            plt.xlabel('number of samples')
            plt.ylabel('estimation error')
            plt.savefig('./results_real/' + name + '/result_plots/' + t + '_all.png')
            plt.close()

        plt.figure()
        for k in results.keys():
            if 'random' in k or k == best_type[file_idx]:
                print(k)
                plt.plot(time,results[k] / result_count[k],label=k)
        plt.legend()
        plt.xlabel('number of samples')
        plt.ylabel('estimation error')
        plt.savefig('./results_real/' + name + '/result_plots/' + t + '_best.png')
        plt.close()

        matplotlib.rcParams.update({'font.size': 40})
        lw = 6
        mw = 6
        ms = 20
        start_idx = 1
        stop_idx = -1
        active_mean = np.mean(file_results_active, axis=0)
        random_mean = np.mean(file_results_passive, axis=0)
        active_std = np.std(file_results_active, axis=0) / np.sqrt(file_results_active.shape[0])
        random_std = np.std(file_results_passive, axis=0) / np.sqrt(file_results_passive.shape[0])
        plt.plot(time[start_idx:stop_idx], active_mean[start_idx:stop_idx],label='Active (Ours)', \
                linewidth=lw,marker='v',markersize=ms,markeredgewidth=mw,color='tab:red')
        plt.fill_between(time[start_idx:stop_idx], active_mean[start_idx:stop_idx] - active_std[start_idx:stop_idx], active_mean[start_idx:stop_idx] + active_std[start_idx:stop_idx],color='tab:red', alpha=.2)
        plt.plot(time[start_idx:stop_idx], random_mean[start_idx:stop_idx],label='Random', \
                linewidth=lw,marker='o',markersize=ms,markeredgewidth=mw,color='tab:blue')
        plt.fill_between(time[start_idx:stop_idx], random_mean[start_idx:stop_idx] - random_std[start_idx:stop_idx], random_mean[start_idx:stop_idx] + random_std[start_idx:stop_idx],color='tab:blue', alpha=.2)
        # plt.errorbar(t,direct_mean,label='Direct Sim2Real Transfer', \
        #         linewidth=lw,marker='o',markersize=ms,markeredgewidth=mw,color='blue',yerr=direct_std)
        # plt.errorbar(t,q_learn_mean,label='Q-Learning with Naive Exploration', \
        #         linewidth=lw,marker='X',markersize=ms,markeredgewidth=mw,color='green',yerr=q_learn_std,linestyle='--')

        plt.xlabel('Number of Trials')
        plt.ylabel('MSE on Heldout Trials')
        if plot_name == '58':
            plt.legend()
        plt.grid()
        fig = plt.gcf()
        fig.set_size_inches(12, 8)
        plt.savefig('./results_real/' + data_name + '_final/best' + str(file_idx) + '.pdf',bbox_inches='tight',format='pdf',dpi=600)

        plt.close()




     #           plt.figure()
      #          plt.
# for k in results.keys():
#     results[k] /= count
print(count)
print(result_count)
print(results_best_count)

# print(np.std(tmp))

# if not os.path.isdir('./results_real/' + name + '/result_plots'):
#     os.makedirs('./results_real/' + name + '/result_plots')


# print(results.keys())



# types = ['random','active']
# for t in types:
#     plt.figure()
#     for k in results.keys():
#         if t in k:
#             plt.plot(time[1:-1],results[k][1:-1] / result_count[k],label=k)
#     plt.legend()
#     plt.xlabel('number of samples')
#     plt.ylabel('estimation error')
#     plt.savefig('./results_real/' + name + '/result_plots/' + t + '_all.png')
#     plt.close()
# quit()


# best = ['active-nrNone-inr100_roc_noin','random-nrNone_roc_noin']
# plt.figure()
# for k in results.keys():
#     if k in best and 'roc_noin' in k:
#         plt.plot(results[k][0] / result_count[k],results[k][1] / result_count[k],label=k)
# plt.legend()
# plt.xlabel('number of samples')
# plt.ylabel('estimation error')
# plt.savefig('./results_real/' + name + '/result_plots/roc_noin_best.png')
# plt.close()



# best = ['active-nrNone-inr100_roc','random-nrNone_roc']
# plt.figure()
# for k in results.keys():
#     if k in best and 'roc' in k:
#         plt.plot(results[k][0] / result_count[k],results[k][1] / result_count[k],label=k)
# plt.legend()
# plt.xlabel('number of samples')
# plt.ylabel('estimation error')
# plt.savefig('./results_real/' + name + '/result_plots/roc_best.png')
# plt.close()







# best = ['active-nrNone-inr100-astart200','active-nrNone-inr100-astart400','random-nrNone']
# plt.figure()
# for k in results.keys():
#     if k in best and 'roc' not in k:
#         plt.plot(time,results[k] / result_count[k],label=k)
# plt.legend()
# plt.xlabel('number of samples')
# plt.ylabel('estimation error')
# plt.savefig('./results_real/' + name + '/result_plots/best.png')
# #plt.savefig('./results_real/best.png')
# plt.close()

# plt.figure()
# for k in results_best.keys():
#     plt.plot(time[1:-1],results_best[k][1:-1] / results_best_count[k],label=k)
# plt.legend()
# plt.xlabel('number of samples')
# plt.ylabel('estimation error')
# #plt.savefig('./results_real/' + name + '/result_plots/best_data0404_071323.png')
# plt.savefig('./results_real/best_' + name[0:6] + '.png')
# plt.close()



# active_best = np.array(results_best_all['active'])
# active_std = np.std(active_best,axis=0) / np.sqrt(results_best_count['active'])
active_best = np.array(std_all_active)
active_std = np.sqrt(np.mean(active_best, axis=0) / results_best_count['active'])
active_mean = results_best['active'] / results_best_count['active']
# random_best = np.array(results_best_all['random'])
# random_std = np.std(random_best,axis=0) / np.sqrt(results_best_count['random'])
random_best = np.array(std_all_random)
random_std = np.sqrt(np.mean(random_best, axis=0) / results_best_count['random'])
print(random_best.shape)
random_mean = results_best['random'] / results_best_count['random']


# matplotlib.rcParams.update({'font.size': 20})
# lw = 3
# mw = 3
# ms = 10
# start_idx = 1
# stop_idx = -1
# plt.plot(time[start_idx:stop_idx], active_mean[start_idx:stop_idx],label='Active (Ours)', \
#         linewidth=lw,marker='v',markersize=ms,markeredgewidth=mw,color='tab:red')
# plt.fill_between(time[start_idx:stop_idx], active_mean[start_idx:stop_idx] - active_std[start_idx:stop_idx], active_mean[start_idx:stop_idx] + active_std[start_idx:stop_idx], alpha=.2,color='tab:red')
# plt.plot(time[start_idx:stop_idx], random_mean[start_idx:stop_idx],label='Random', \
#         linewidth=lw,marker='o',markersize=ms,markeredgewidth=mw,color='tab:blue')
# plt.fill_between(time[start_idx:stop_idx], random_mean[start_idx:stop_idx] - random_std[start_idx:stop_idx], random_mean[start_idx:stop_idx] + random_std[start_idx:stop_idx], alpha=.2,color='tab:blue')
# # plt.errorbar(t,direct_mean,label='Direct Sim2Real Transfer', \
# #         linewidth=lw,marker='o',markersize=ms,markeredgewidth=mw,color='blue',yerr=direct_std)
# # plt.errorbar(t,q_learn_mean,label='Q-Learning with Naive Exploration', \
# #         linewidth=lw,marker='X',markersize=ms,markeredgewidth=mw,color='green',yerr=q_learn_std,linestyle='--')

# plt.xlabel('Number of Trajectories')
# plt.ylabel('MSE on Heldout Inputs')
# plt.legend()
# plt.grid()
# fig = plt.gcf()
# fig.set_size_inches(6, 4)
# plt.savefig('./results_real/real_' + plot_name + '.pdf',transparent=True,bbox_inches='tight',format='pdf',dpi=600)
 
# plt.close()




matplotlib.rcParams.update({'font.size': 40})
lw = 6
mw = 6
ms = 20
start_idx = 1
stop_idx = -1
plt.plot(time[start_idx:stop_idx], active_mean[start_idx:stop_idx],label='Active (Ours)', \
        linewidth=lw,marker='v',markersize=ms,markeredgewidth=mw,color='tab:red')
plt.fill_between(time[start_idx:stop_idx], active_mean[start_idx:stop_idx] - active_std[start_idx:stop_idx], active_mean[start_idx:stop_idx] + active_std[start_idx:stop_idx], alpha=.2,color='tab:red')
plt.plot(time[start_idx:stop_idx], random_mean[start_idx:stop_idx],label='Random', \
        linewidth=lw,marker='o',markersize=ms,markeredgewidth=mw,color='tab:blue')
plt.fill_between(time[start_idx:stop_idx], random_mean[start_idx:stop_idx] - random_std[start_idx:stop_idx], random_mean[start_idx:stop_idx] + random_std[start_idx:stop_idx], alpha=.2,color='tab:blue')
# plt.errorbar(t,direct_mean,label='Direct Sim2Real Transfer', \
#         linewidth=lw,marker='o',markersize=ms,markeredgewidth=mw,color='blue',yerr=direct_std)
# plt.errorbar(t,q_learn_mean,label='Q-Learning with Naive Exploration', \
#         linewidth=lw,marker='X',markersize=ms,markeredgewidth=mw,color='green',yerr=q_learn_std,linestyle='--')

plt.xlabel('Number of Trials')
plt.ylabel('MSE on Heldout Trials')
if plot_name == '58':
    plt.legend()
plt.grid()
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.savefig('./results_real/real_' + plot_name + '.pdf',transparent=True,bbox_inches='tight',format='pdf',dpi=600)
 
plt.close()





