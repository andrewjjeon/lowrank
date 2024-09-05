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


data_id = int(sys.argv[1])
rank = int(sys.argv[2])

if data_id == 60:
    best = ['active-nr25-inr25-0','random-nr25-0','uniform-nr25-0']
    if rank == 35:
        name = 'photostim_60_r35'
        mat_norm = 8.26282631057816
    elif rank == 15:
        name = 'photostim_60_r15'
        mat_norm = 7.19970461053275
elif data_id == 59:
    best = ['active-nr5-inr75-0','random-nr5-0','uniform-nr5-0']
    if rank == 35:
        name = 'photostim_59_r35'
        mat_norm = 2.649534920571048
    elif rank == 15:
        name = 'photostim_59_r15'
        mat_norm = 2.5747274332336714
elif data_id == 58:
    if rank == 35:
        name = 'photostim_58_r35_long'
        best = ['active-nr10-inr10-0', 'random-nr10-0', 'uniform-nr10-0']
        mat_norm = 3.4783367569471175
    elif rank == 15:
        name = 'photostim_58_r15_long'
        best = ['active-nr5-inr10-0','random-nr5-0','uniform-nr5-0']
        mat_norm = 2.979932886109933
elif data_id == 663:
    if rank == 35:
        name = 'd663_lr35_final'
        best = ['random-nr100-0','active-nr100-inr50-0','oracle-nr100-inr50-0','uniform-nr100-0']
        mat_norm = 18.247240818268406
    elif rank == 15:
        name = 'd663_lr15_final'
        best = ['random-nr75-0','active-nr75-inr25-0','oracle-nr75-inr25-0','uniform-nr75-0']
        mat_norm = 18.00834552692588





#name = 'photostim_60_r35'
#name = 'photostim_0404_date_070623_r15_sweep_noise0.4'
#name = 'photostim_58_r35'
files = os.listdir('./results/' + name)
print(files)

results = {}
results_all = {}
count = 0
time = None

for i in range(len(files)):
    if 'results' in files[i]:
        with open('./results/' + name + '/' + files[i], 'rb') as f:
            x = pickle.load(f)
            print(x[1])
            count += 1
            for k in x[0].keys():
                if k in results.keys():
                    results[k] += np.array(x[0][k]['nuc_nodiag']) / mat_norm
                    results_all[k].append(np.array(x[0][k]['nuc_nodiag']) / mat_norm)
                    #results[k + '_ls'] += np.array(x[0][k]['ls_nodiag'])
                else:
                    results[k] = np.array(x[0][k]['nuc_nodiag']) / mat_norm
                    results_all[k] = [np.array(x[0][k]['nuc_nodiag']) / mat_norm]
                    #results[k + '_ls'] = np.array(x[0][k]['ls_nodiag'])
                time = x[0][k]['time']
    if count == 20:
        break
print('count = ' + str(count))
for k in results.keys():
    results[k] /= count

if not os.path.isdir('./results/' + name + '/result_plots'):
    os.makedirs('./results/' + name + '/result_plots')


types = ['random','active','uniform','oracle']
for t in types:
    plt.figure()
    count2 = 0
    for k in results.keys():
        if t in k:
            if count2 > 9:
                plt.plot(time,results[k],'--',label=k)
            else:
                plt.plot(time,results[k],label=k)
            count2 += 1
    plt.legend(loc='upper right')
    plt.xlabel('number of samples')
    plt.ylabel('estimation error')
    plt.savefig('./results/' + name + '/result_plots/' + t + '_all.png')
    plt.close()


# best for d663_lr35
# best = ['random-nr100-0','active-nr100-inr50-0','oracle-nr100-inr50-0','uniform-nr100-0']

# best for d663_lr15
#best = ['random-nr75-0','active-nr75-inr25-0','oracle-nr75-inr25-0','uniform-nr75-0']

# best for photostim_0404_date_070623_r15_sweep
#best = ['random-nr75-0','uniform-nr75-0','oracle-nr75-inr50-0','active-nr75-inr25-0']

# best for photostim_0404_date_070623_r15_sweep_noise0.4
#best = ['random-nr50-0','uniform-nr50-0','oracle-nr50-inr50-0','active-nr50-inr25-0']

# best for photostim_0404_date_070623_r15_sweep_long_halfrand
#best = ['random-nr75-0','active-nr50-inr50-0','active-nr75-inr50-0']

# best for photostim_0404_date_070623_r15_final
#best = ['active-nr50-inr25-0','oracle-nr50-inr75-0','random-nr50-0','uniform-nr50-0']

# best for photostim_0404_date_070623_r35_final
# best = ['active-nr100-inr50-0','oracle-nr50-inr50-0','random-nr100-0','uniform-nr100-0']

# best for photostim_0404_date_070623_r15_std0.5_2

# best for photostim_58_r15 (set stop idx to -10)
#best = ['active-nr5-inr10-0','random-nr5-0','uniform-nr5-0']

# best for photostim_58_r35 (set stop idx to -10)
#best = ['active-nr10-inr10-0', 'random-nr10-0', 'uniform-nr10-0']

# best for photostim_59_r15/r35
#best = ['active-nr5-inr75-0','random-nr5-0','uniform-nr5-0']

# best for photostim_60_r15/r35
# best = ['active-nr25-inr25-0','random-nr25-0','uniform-nr25-0']



results_std = {}
for k in results_all.keys():
    temp = np.array(results_all[k])
    results_std[k] = np.std(temp, axis=0) / np.sqrt(temp.shape[0])



plt.figure()
for k in results.keys():
    if k in best:
        plt.plot(time,results[k],label=k)
        #plt.plot(time,results[k + '_ls'],label=k + ' (ls)')
plt.legend()
plt.xlabel('number of samples')
plt.ylabel('estimation error')
plt.savefig('./results/' + name + '/result_plots/best.png')
plt.close()



matplotlib.rcParams.update({'font.size': 40})
lw = 6
mw = 6
ms = 20
start_idx = 0
stop_idx = -10

if data_id == 58:
    for k in results.keys():
        if k in best:
            if 'active' in k:
                plt.plot(time[start_idx:stop_idx], results[k][start_idx:stop_idx],label='Active (Ours)', \
                        linewidth=lw,marker='v',markersize=ms,markeredgewidth=mw,color='tab:red')
                plt.fill_between(time[start_idx:stop_idx], results[k][start_idx:stop_idx] - results_std[k][start_idx:stop_idx], results[k][start_idx:stop_idx] + results_std[k][start_idx:stop_idx],color='tab:red', alpha=.2)
            if 'random' in k: 
                plt.plot(time[start_idx:stop_idx], results[k][start_idx:stop_idx],label='Random', \
                        linewidth=lw,marker='o',markersize=ms,markeredgewidth=mw,color='tab:blue')
                plt.fill_between(time[start_idx:stop_idx], results[k][start_idx:stop_idx] - results_std[k][start_idx:stop_idx], results[k][start_idx:stop_idx] + results_std[k][start_idx:stop_idx],color='tab:blue', alpha=.2)
            if 'uniform' in k: 
                plt.plot(time[start_idx:stop_idx], results[k][start_idx:stop_idx],label='Uniform', \
                        linewidth=lw,marker='X',markersize=ms,markeredgewidth=mw,color='tab:green')
                plt.fill_between(time[start_idx:stop_idx], results[k][start_idx:stop_idx] - results_std[k][start_idx:stop_idx], results[k][start_idx:stop_idx] + results_std[k][start_idx:stop_idx],color='tab:green', alpha=.2)
else:
    for k in results.keys():
        if k in best:
            if 'active' in k:
                plt.plot(time[start_idx:], results[k][start_idx:],label='Active (Ours)', \
                        linewidth=lw,marker='v',markersize=ms,markeredgewidth=mw,color='tab:red')
                plt.fill_between(time[start_idx:], results[k][start_idx:] - results_std[k][start_idx:], results[k][start_idx:] + results_std[k][start_idx:],color='tab:red', alpha=.2)
            if 'random' in k: 
                plt.plot(time[start_idx:], results[k][start_idx:],label='Random', \
                        linewidth=lw,marker='o',markersize=ms,markeredgewidth=mw,color='tab:blue')
                plt.fill_between(time[start_idx:], results[k][start_idx:] - results_std[k][start_idx:], results[k][start_idx:] + results_std[k][start_idx:],color='tab:blue', alpha=.2)
            if 'uniform' in k: 
                plt.plot(time[start_idx:], results[k][start_idx:],label='Uniform', \
                        linewidth=lw,marker='X',markersize=ms,markeredgewidth=mw,color='tab:green')
                plt.fill_between(time[start_idx:], results[k][start_idx:] - results_std[k][start_idx:], results[k][start_idx:] + results_std[k][start_idx:],color='tab:green', alpha=.2)



# plt.errorbar(t,direct_mean,label='Direct Sim2Real Transfer', \
#         linewidth=lw,marker='o',markersize=ms,markeredgewidth=mw,color='blue',yerr=direct_std)
# plt.errorbar(t,q_learn_mean,label='Q-Learning with Naive Exploration', \
#         linewidth=lw,marker='X',markersize=ms,markeredgewidth=mw,color='green',yerr=q_learn_std,linestyle='--')

#plt.yscale('log')
plt.xlabel('Number of Trials')
plt.ylabel('Estimation Error')
if '58' in name:
    plt.legend()
plt.grid()
fig = plt.gcf()
fig.set_size_inches(12, 8)
plt.savefig('./results/learned_' + str(data_id) + '_r' + str(rank) + '.pdf',bbox_inches='tight',format='pdf',dpi=600,transparent=True)

plt.close()





