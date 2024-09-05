import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.set_default_device('cpu')
    
    
def design_inputs_V(transfer_est, num_batches=1000, verbose=False, n_iters=1000):
    u_batch = torch.tensor(torch.randn(d,num_batches), requires_grad=True, dtype=torch.float32)
    with torch.no_grad():
        #u_batch.clamp_(min=0,max=1)
        u_batch.data = np.sqrt(num_batches) * u_batch / torch.linalg.norm(u_batch.clone().detach(), ord='fro')
    optimizer = optim.Adam([u_batch], lr=0.05)

    num_random = int(T/2) #num_batches
    u_random = np.random.rand(d,num_random)
    u_random = np.sqrt(num_random) * u_random / np.linalg.norm(u_random, 'fro')
    #u_random = u_random > 0.98
    u_random = torch.tensor(u_random, dtype=torch.float32)
    #u_random = project_inputs(u_random)
    #u_random.clamp_(min=0,max=1)
    cov_random = u_random @ u_random.T / num_random
    
#     u_random = np.random.rand(d,num_random)
#     u_random = u_random > 0.4
#     u_random = torch.tensor(u_random, dtype=torch.float32)
#     u_random = project_inputs(u_random)
#     u_random.clamp_(min=0,max=1)
#     cov_random = u_random @ u_random.T / num_batches

    _,S_true,V_true = la.svd(transfer_est)
#     V_true2 = []
#     max_S = np.max(S_true)
#     for i in range(len(S_true)):
#         if S_true[i] > 0.01*max_S:
#             V_true2.append(V_true[i])
#     V_true = torch.tensor(V_true2, dtype=torch.float32)
    V_true = torch.tensor(V_true[0:100,:], dtype=torch.float32)
    #print(V_true.shape)
    
    loss_vals = []
    reg = 50 # previously 50, 0.01
    vst_torch = torch.tensor(vst).float()
    for n in range(n_iters):
        optimizer.zero_grad()
        cov = u_batch @ u_batch.T / num_batches
        #cov_random = 0
        #loss = 1 / (vst_torch.T @ (cov_random + cov + 0.000001*torch.eye(d)) @ vst_torch)
        loss = torch.trace(torch.linalg.inv( V_true @ (cov_random + cov + 0.000001*torch.eye(d)) @ V_true.T))
        #loss += torch.trace(torch.linalg.inv(cov_random + cov + 0.000001*torch.eye(d)))
        #l1_norms = torch.linalg.norm(u_batch, 1, axis=0)
        #max_norm = torch.logsumexp(l1_norms, 0)
        #loss += reg * max_norm
        #loss += reg * torch.linalg.norm(u_batch, 1)
        loss_val = loss.item()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            #u_batch.clamp_(min=0,max=1)
            if torch.linalg.norm(u_batch.clone().detach(), ord='fro') > np.sqrt(num_batches):
                u_batch.data = np.sqrt(num_batches) * u_batch / torch.linalg.norm(u_batch.clone().detach(), ord='fro')
        if np.mod(n,200) == 0 and verbose:
            #nnz = torch.sum(u_batch > 0.01)
            #print(n,loss_val,nnz/num_batches)
            nnz_column = torch.sum(u_batch > 0.01, axis=0)
            print(n,loss_val,torch.max(nnz_column))
        loss_vals.append(loss_val)
        
    nnz_column = torch.sum(u_batch > 0.01, axis=0)
    print(n,loss_val,torch.max(nnz_column))
    plt.plot(loss_vals[10:])
    plt.show()
        
    u_batch_np = u_batch.clone().detach().numpy()
    u_random_np = u_random.clone().detach().numpy()

    return u_batch_np, u_random_np





def project_inputs(u_batch, max_on=15):
    batch_size = u_batch.shape[1]
    for i in range(batch_size):
        sort_idx = torch.argsort(u_batch[:,i], descending=True)
        u_batch[sort_idx[max_on:],i] = 0.0
        u_batch[sort_idx[:max_on],i] = 1.0
    return u_batch

def compute_inverse_trace(transfer,r,inputs,reg_cov):
    U,S,V = torch.linalg.svd(transfer)
    V0 = V[0:r,:].T
    V1 = V[r:,:].T
    Sigma = inputs @ inputs.T + reg_cov
    
    VSig00inv = torch.linalg.inv(V0.T @ Sigma @ V0)
    Dinv = torch.linalg.inv(V1.T @ Sigma @ V1 - V1.T @ Sigma @ V0 @ VSig00inv @ V0.T @ Sigma @ V1)
    A1inv = VSig00inv @ V0.T @ Sigma @ V1 @ Dinv @ V1.T @ Sigma @ V0 @ VSig00inv
    return ((d-r)*torch.trace(VSig00inv) + r*torch.trace(VSig00inv + A1inv) + r*torch.trace(Dinv))/d

def design_inputs_fisher(transfer_est, num_batches=1000, verbose=True, n_iters=1000, sphere_normalization=False, num_random=1000):
    d = transfer_est.shape[0]
    u_batch = torch.tensor(0.1*torch.randn(d,num_batches), requires_grad=True, dtype=torch.float32)
    with torch.no_grad():
        if sphere_normalization:
            u_batch.data = np.sqrt(num_batches) * u_batch / torch.linalg.norm(u_batch.clone().detach(), ord='fro')
        else:
            u_batch.clamp_(min=0,max=1)
            u_batch = project_inputs(u_batch, max_on=10)
    #optimizer = optim.Adam([u_batch], lr=0.025)
    optimizer = optim.Adam([u_batch], lr=0.5)
    
    # num_random = int(T/2)
    u_random = np.random.randn(d,num_random)
    if sphere_normalization:
        u_random = np.sqrt(num_random) * u_random / np.linalg.norm(u_random, 'fro')
    #u_random = u_random > 0.4
    u_random = torch.tensor(u_random, dtype=torch.float32)
    if not sphere_normalization:
        u_random = project_inputs(u_random, max_on=14)
        u_random.clamp_(min=0,max=1)

    U_true,S_true,V_true = torch.linalg.svd(transfer_est)
    k = 100
    loss_vals = []
    reg = 1500 # 700 was working when cov_reg = 0.00001 * torch.eye(d), 2700 when using random reg
    reg2 = 10 # 0.25
    for n in range(n_iters):
        optimizer.zero_grad()
        #cov_reg = u_random @ u_random.T / num_random
        #if num_random < d:
        cov_reg = 0.000001 * torch.eye(d)
        loss = compute_inverse_trace(transfer_est,k,u_batch / np.sqrt(num_batches), cov_reg)
        loss_exp = loss.item()
        if not sphere_normalization:
            #l1_norms = torch.linalg.norm(u_batch, 1, axis=0)
            #max_norm = torch.logsumexp(l1_norms, 0)
            #loss += reg * max_norm
            loss += reg2 * torch.linalg.norm(u_batch.flatten(), 1)
        loss_val = loss.item()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            if sphere_normalization:
                if torch.linalg.norm(u_batch.clone().detach(), ord='fro') > np.sqrt(num_batches):
                    u_batch.data = np.sqrt(num_batches) * u_batch / torch.linalg.norm(u_batch.clone().detach(), ord='fro')
            else:
                u_batch.clamp_(min=0,max=1)
        if np.mod(n,5) == 0 and verbose:
            nnz_column = torch.sum(u_batch > 0.01, axis=0)
            print(n,loss_val,loss_exp,torch.mean(nnz_column.float()).numpy(),la.norm(u_batch.clone().detach().flatten().numpy(), 1)/num_batches)
            if torch.norm(u_batch.clone().detach(), 1)/num_batches < 14.5 and n > 50:
                break
        loss_vals.append(loss_val)
    
    plt.plot(loss_vals)
    plt.show()
    u_batch_np = u_batch.clone().detach().numpy()
    u_random_np = u_random.clone().detach().numpy()
    return u_batch_np, u_random_np






def l1_proj(u_batch, max_on=15):
    d,n = u_batch.shape
    u_batch.clamp_(min=0, max=1)
    u_proj = torch.zeros_like(u_batch)
    for i in range(n):
        sort_idx = torch.argsort(u_batch[:,i], descending=True)
        running_sum = 0
        max_j = 0
        max_sum = 0
        for j in range(d):
            uij = u_batch[sort_idx[j],i]
            running_sum += uij
            if uij - (running_sum - max_on) / (j+1) > 0:
                max_j = j + 1
                max_sum = running_sum.clone()
        theta = (max_sum - max_on) / max_j
        u_proj[:,i] = u_batch[:,i] - theta
        u_proj[:,i].clamp_(min=0)
    u_batch.data = u_proj
    return u_batch

def l1_proj_vec(u_batch, max_on=15):
    d,n = u_batch.shape
    batch_idx = torch.linspace(0,n-1,n).int()
    u_batch.clamp_(min=0, max=1)
    u_proj = torch.zeros_like(u_batch)
    sort_idx = torch.argsort(u_batch, dim=0, descending=True)
    running_sum = torch.zeros(n)
    max_j = torch.zeros(n)
    max_sum = torch.zeros(n)
    for j in range(d):
        uj = u_batch[sort_idx[j,:], torch.arange(n)]
        running_sum += uj
        good = uj - (running_sum - max_on) / (j+1) > 0
        max_j[good] = j+1
        max_sum[good] = running_sum[good].clone()    
    theta = torch.divide(max_sum - max_on,max_j)
    u_proj = u_batch - torch.outer(torch.ones(d),theta)
    u_proj.clamp_(min=0)
    u_batch.data = u_proj
    return u_batch

def project_inputs(u_batch, max_on=15):
    batch_size = u_batch.shape[1]
    for i in range(batch_size):
        sort_idx = torch.argsort(u_batch[:,i], descending=True)
        u_batch[sort_idx[max_on:],i] = 0.0
        u_batch[sort_idx[:max_on],i] = 1.0
    return u_batch

def est_V(U,Y,r,percent=0.25):
    Y_norms = []
    for i in range(len(Y)):
        Y_norms.append(np.linalg.norm(Y[i], 2))
    order = np.argsort(Y_norms)
    num_keep = int(percent * len(Y_norms))
    X = np.zeros((U[0].shape[0], num_keep))
    for i in range(num_keep):
        X[:,i] = U[order[-i]]
    V,S,_ = np.linalg.svd(X)
    plt.plot(S)
    plt.show()
    return V[:,0:r].T

def design_inputs_constrained(transfer_est, 
        num_batches=1000, 
        verbose=True, 
        n_iters=1000, 
        l1_constraint=15, 
        k=50, 
        sphere_normalization=False, 
        num_random=None,
        V_design=True,
        cov0=None,
        V_true=None,
        plt_save=None):
    transfer_est = torch.tensor(transfer_est).float()
    d = transfer_est.shape[0]
    u_batch = torch.tensor(0.1*torch.randn(d,num_batches), requires_grad=True, dtype=torch.float32)
    with torch.no_grad():
        u_batch = l1_proj_vec(u_batch, max_on=l1_constraint)
    optimizer = optim.Adam([u_batch], lr=0.1)
    
    if num_random is not None:
        u_random = np.random.randn(d,num_random)
        if sphere_normalization:
            u_random = np.sqrt(num_random) * u_random / np.linalg.norm(u_random, 'fro')
        u_random = torch.tensor(u_random, dtype=torch.float32)
        if not sphere_normalization:
            u_random = project_inputs(u_random, max_on=l1_constraint)
            u_random.clamp_(min=0,max=1)
        cov_random = u_random @ u_random.T / num_random
    if cov0 is not None:
        cov0 = torch.tensor(cov0).float()

    if V_true is None:
        _,_,V_true = torch.linalg.svd(transfer_est)
        V_true = torch.tensor(V_true[0:k,:], dtype=torch.float32)
    else:
        V_true = torch.tensor(V_true, dtype=torch.float32)
    min_inputs = None
    min_loss = 1e10
    loss_vals = []
    
    for n in range(n_iters):
        optimizer.zero_grad()
        cov_reg = 0.000001 * torch.eye(d)
        if num_random is not None:
            cov_reg += cov_random
        if cov0 is not None:
            cov_reg += cov0
        if V_design:
            cov = u_batch @ u_batch.T / num_batches
            loss = torch.trace(torch.linalg.inv( V_true @ (cov + cov_reg) @ V_true.T))
        else:
            cov_reg += cov_batch1
            loss = compute_inverse_trace(transfer_est,k,u_batch / np.sqrt(num_batches), cov_reg)
        loss_val = loss.item()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            u_batch = l1_proj_vec(u_batch, max_on=l1_constraint)
        if np.mod(n,10) == 0 and verbose:
            nnz_column = torch.sum(u_batch > 0.01, axis=0)
            print(n,loss_val,torch.mean(nnz_column.float()).numpy(),la.norm(u_batch.clone().detach().flatten().numpy(), 1)/num_batches)
        loss_vals.append(loss_val)
        if loss_val < min_loss:
            min_loss = loss_val
            min_inputs = u_batch.clone()
    #if verbose:
    plt.plot(loss_vals)
    plt.title("input optimization loss")
    plt.show()
    if plt_save is not None:
        plt.savefig(plt_save + ".png")
        plt.close()

    u_batch_np = min_inputs.clone().detach().numpy()
    if num_random is not None:
        u_random_np = u_random.clone().detach().numpy()
    else:
        u_random_np = None
    return u_batch_np, u_random_np


def design_inputs_constrained_weighted(transfer_est, 
        num_batches=1000, 
        verbose=True, 
        n_iters=1000, 
        l1_constraint=15, 
        k=50, 
        sphere_normalization=False, 
        num_random=None,
        V_design=True,
        cov0=None):
    transfer_est = torch.tensor(transfer_est).float()
    d = transfer_est.shape[0]
    u_batch = torch.tensor(0.1*torch.randn(d,num_batches), requires_grad=True, dtype=torch.float32)
    with torch.no_grad():
        u_batch = l1_proj_vec(u_batch, max_on=l1_constraint)
    optimizer = optim.Adam([u_batch], lr=0.1)
    
    if num_random is not None:
        u_random = np.random.randn(d,num_random)
        if sphere_normalization:
            u_random = np.sqrt(num_random) * u_random / np.linalg.norm(u_random, 'fro')
        u_random = torch.tensor(u_random, dtype=torch.float32)
        if not sphere_normalization:
            u_random = project_inputs(u_random, max_on=l1_constraint)
            u_random.clamp_(min=0,max=1)
        cov_random = u_random @ u_random.T / num_random
    if cov0 is not None:
        cov0 = torch.tensor(cov0).float()

    _,S_true,V_true = torch.linalg.svd(transfer_est)
    V_true = torch.diag(torch.sqrt(S_true)) @ V_true / torch.max(torch.sqrt(S_true))
    min_inputs = None
    min_loss = 1e10
    loss_vals = []
    
    for n in range(n_iters):
        optimizer.zero_grad()
        cov_reg = 0.000001 * torch.eye(d)
        if num_random is not None:
            cov_reg += cov_random
        if cov0 is not None:
            cov_reg += cov0
        if V_design:
            cov = u_batch @ u_batch.T / num_batches
            loss = torch.trace(torch.linalg.inv( V_true @ (cov + cov_reg) @ V_true.T))
        else:
            cov_reg += cov_batch1
            loss = compute_inverse_trace(transfer_est,k,u_batch / np.sqrt(num_batches), cov_reg)
        loss_val = loss.item()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            u_batch = l1_proj_vec(u_batch, max_on=l1_constraint)
        if np.mod(n,10) == 0 and verbose:
            nnz_column = torch.sum(u_batch > 0.01, axis=0)
            print(n,loss_val,torch.mean(nnz_column.float()).numpy(),la.norm(u_batch.clone().detach().flatten().numpy(), 1)/num_batches)
        loss_vals.append(loss_val)
        if loss_val < min_loss or n == 0:
            min_loss = loss_val
            min_inputs = u_batch.clone()
    #if verbose:
    plt.plot(loss_vals)
    plt.title("input optimization loss")
    plt.show()

    u_batch_np = min_inputs.clone().detach().numpy()
    if num_random is not None:
        u_random_np = u_random.clone().detach().numpy()
    else:
        u_random_np = None
    return u_batch_np, u_random_np