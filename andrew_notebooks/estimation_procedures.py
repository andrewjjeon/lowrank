import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#torch.set_default_device('cuda:0')


class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, device='cuda:0').float()
        self.Y = torch.tensor(Y, device='cuda:0').float()
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        X_samp = self.X[idx,:]
        Y_samp = self.Y[idx,:]
        return X_samp, Y_samp


def estimate(X,Y):
    X = np.array(X)
    Y = np.array(Y)
    U,S,V = la.svd(Y.T)
    return U[:,0]

def estimate_ls(X,Y):
    X = np.array(X)
    Y = np.array(Y)
    thetahat = la.pinv(X.T @ X + 0.001 * np.eye(X.shape[1])) @ X.T @ Y
    #U,S,_ = la.svd(thetahat)
    return thetahat.T #U[:,:k] @ np.diag(np.sqrt(S[:k]))

def estimate_gd(X,Y,n_iters=None):
#     X = torch.tensor(X).float()
#     Y = torch.tensor(Y).float()
    data = CustomDataset(X,Y)
    batch_size = 128
    shuffle = True
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, drop_last=len(X)>batch_size)

    #print('starting gd')
    #u_hat2 = estimate_ls(X,Y)
    u_hat2 = 0.1*np.random.randn(d,k)
    u_hat = torch.tensor(u_hat2).float()
    u_hat.requires_grad_()
    optimizer = optim.SGD([u_hat], lr=0.01)
    loss_vals = []
    total_steps = 10000
    n_max = np.max([int(10000 / len(X)), 1000])
    if n_iters is not None:
        n_max = int(n_iters / len(X))
        if n_max < 1:
            n_max = 1
    for n in range(n_max):
        for x_batch, y_batch in data_loader:
            optimizer.zero_grad()
            loss = torch.linalg.norm(y_batch.T - u_hat @ u_hat.T @ x_batch.T, ord='fro')**2
            loss_val = loss.item()
            loss.backward()
            optimizer.step()
            loss_vals.append(loss_val)
        #if np.mod(n,100) == 0:
        #    print(loss_val)
    if n_iters is None:
        plt.plot(loss_vals)
        plt.show()
    return u_hat.detach().numpy()


def estimate_gd_nuc(X,Y,n_iters=None,reg=0.00001,lr=None,step_multiplier=None):
    data = CustomDataset(X,Y)
    batch_size = 512
    TX = len(X)
    X = torch.tensor(X, device='cuda:0').float()
    _,d = X.shape
    Y = torch.tensor(Y, device='cuda:0').float()
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, drop_last=len(X)>batch_size, generator=torch.Generator(device='cuda'))

    #u_hat2 = estimate_ls(X,Y)
    u_hat2 = 0.1*np.random.randn(d,d)
    u_hat = torch.tensor(u_hat2, device='cuda:0').float()
    u_hat.requires_grad_()
    if lr is None:
        optimizer = optim.SGD([u_hat], lr=0.05)
    else:
        optimizer = optim.SGD([u_hat], lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    loss_vals = []
    if step_multiplier is None:
        total_steps = int(300000/5)
    else:
        total_steps = int(step_multiplier*300000/5)
    n_max = np.max([int(total_steps / len(X)), 10])
    if n_iters is not None:
        n_max = int(n_iters / len(X))
        if n_max < 1:
            n_max = 1
    #n_max = 100
    for n in range(n_max):
        #loss = 1/TX*torch.linalg.norm(Y.T - u_hat @ X.T, ord='fro')**2
        #loss += 0.0001*torch.linalg.norm(u_hat, ord='fro')**2
        #loss += reg*torch.linalg.norm(u_hat, ord='nuc')
#         loss_val = loss.item()
#         loss.backward()
#         optimizer.step()
#         loss_vals.append(loss_val)
        for x_batch, y_batch in data_loader:
            optimizer.zero_grad()
            loss = torch.linalg.norm(y_batch.T - u_hat @ x_batch.T, ord='fro')**2
            loss += reg*torch.linalg.norm(u_hat, ord='nuc')
            loss_val = loss.item()
            loss.backward()
            optimizer.step()
            loss_vals.append(loss_val)
        scheduler.step()
        #print(scheduler.get_last_lr())
    if n_iters is None:
        plt.plot(loss_vals)
        plt.show()
    return u_hat.detach().cpu().numpy()


def l1_proj(u_batch, max_on=15):
    if torch.linalg.norm(u_batch, 1) <= max_on:
        return u_batch
    d = u_batch.shape[0]
    u_proj = torch.zeros_like(u_batch)
    sort_idx = torch.argsort(u_batch, descending=True)
    running_sum = 0
    max_j = 0
    max_sum = 0
    for j in range(d):
        uj = u_batch[sort_idx[j]]
        running_sum += uj
        if uj - (running_sum - max_on) / (j+1) > 0:
            max_j = j + 1
            max_sum = running_sum.clone()
    theta = (max_sum - max_on) / max_j
    u_proj = u_batch - theta
    u_proj.clamp_(min=0)
    return u_proj


def estimate_gd_nuc_project(X,Y,reg=0.00001,lr=None,n_iters=1000,transfer0=None,plt_save=None):
    data = CustomDataset(X,Y)
    batch_size = 512
    X = torch.tensor(X, device='cuda:0').float()
    _,d = X.shape
    Y = torch.tensor(Y, device='cuda:0').float()
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, drop_last=len(X)>batch_size, generator=torch.Generator(device='cuda'))

    if transfer0 is None:
        #u_hat2 = estimate_ls(X,Y)
        u_hat2 = 0.1*np.random.randn(d,d)
    else:
        u_hat2 = transfer0
    u_hat = torch.tensor(u_hat2, device='cuda:0').float()
    U,S,V = torch.linalg.svd(u_hat)
    S = l1_proj(S, max_on=reg)
    u_hat.data = U @ torch.diag(S) @ V
    u_hat.requires_grad_()
    if lr is None:
        optimizer = optim.SGD([u_hat], lr=0.05)
    else:
        optimizer = optim.SGD([u_hat], lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    loss_vals = []
    for n in range(n_iters):
        #for x_batch, y_batch in data_loader:
        optimizer.zero_grad()
        #loss = torch.linalg.norm(y_batch.T - u_hat @ x_batch.T, ord='fro')**2
        loss = torch.linalg.norm(Y.T - u_hat @ X.T, ord='fro')**2
        loss_val = loss.item()
        loss.backward()
        optimizer.step()
        loss_vals.append(loss_val)
        with torch.no_grad():
            U,S,V = torch.linalg.svd(u_hat)
            S = l1_proj(S, max_on=reg)
            u_hat.data = U @ torch.diag(S) @ V
        #scheduler.step()
        #print(scheduler.get_last_lr())
        # if np.mod(n,10) == 0:
        #     print(n,loss_val,torch.linalg.norm(u_hat, 'nuc'))
    plt.plot(loss_vals)
    plt.title("nuc estimation loss")
    plt.show()
    if plt_save is not None:
        plt.savefig(plt_save + ".png")
        plt.close()
    return u_hat.detach().cpu().numpy()


def estimate_gd_nuc_project_AB(X,Y,reg=0.00001,lr=0.0001,n_iters=1000,transfer0=None):
    Y = torch.tensor(Y, device='cuda:0').float()
    n_pts,d = Y.shape
    X = torch.tensor(X, device='cuda:0').float()
    X_B = X[:,0:d]
    X_A = X[:,d:]

    if transfer0 is None:
        A_hat = 0.1*np.random.randn(d,d)
        B_hat = 0.1*np.random.randn(d,d)
    else:
        A_hat = transfer0[0]
        B_hat = transfer0[1]
    A_hat = torch.tensor(A_hat, device='cuda:0').float()
    B_hat = torch.tensor(A_hat, device='cuda:0').float()
    U,S,V = torch.linalg.svd(B_hat)
    S = l1_proj(S, max_on=reg)
    B_hat.data = U @ torch.diag(S) @ V
    A_hat.requires_grad_()
    B_hat.requires_grad_()
    
    optimizer = optim.SGD([A_hat,B_hat], lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    loss_vals = []
    for n in range(n_iters):
        optimizer.zero_grad()
        loss = torch.linalg.norm(Y.T - A_hat @ X_A.T - B_hat @ X_B.T, ord='fro')**2 / n_pts
        loss += 0.00001 * torch.linalg.norm(A_hat, ord='fro')**2 + 0.00001 * torch.linalg.norm(B_hat, ord='fro')**2
        loss_val = loss.item()
        loss.backward()
        optimizer.step()
        loss_vals.append(loss_val)
        with torch.no_grad():
            U,S,V = torch.linalg.svd(B_hat)
            S = l1_proj(S, max_on=reg)
            B_hat.data = U @ torch.diag(S) @ V
        # scheduler.step()
    plt.plot(loss_vals)
    plt.title("nuc estimation loss")
    plt.show()
    return A_hat.detach().cpu().numpy(), B_hat.detach().cpu().numpy()


def estimate_gd_nuc_project_AB_offset(X,Y,reg=0.00001,lr=0.0001,n_iters=1000,transfer0=None,plt_save=None):
    Y = torch.tensor(Y, device='cuda:0').float()
    n_pts,d = Y.shape
    X = torch.tensor(X, device='cuda:0').float()
    X_B = X[:,0:d]
    X_A = X[:,d:]

    if transfer0 is None:
        A_hat = 0.1*np.random.randn(d,d)
        B_hat = 0.1*np.random.randn(d,d)
        v_hat = 0.1*np.random.randn(d)
    else:
        A_hat = transfer0[0]
        B_hat = transfer0[1]
        v_hat = transfer0[2]
    A_hat = torch.tensor(A_hat, device='cuda:0').float()
    B_hat = torch.tensor(B_hat, device='cuda:0').float()
    v_hat = torch.tensor(v_hat, device='cuda:0').float()
    U,S,V = torch.linalg.svd(B_hat)
    S = l1_proj(S, max_on=reg)
    B_hat.data = U @ torch.diag(S) @ V
    A_hat.requires_grad_()
    B_hat.requires_grad_()
    v_hat.requires_grad_()
    
    optimizer = optim.SGD([A_hat,B_hat,v_hat], lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    loss_vals = []
    #reg = 0.00001
    for n in range(n_iters):
        optimizer.zero_grad()
        _,S,_ = torch.linalg.svd(B_hat)
        loss = torch.linalg.norm(Y.T - A_hat @ X_A.T - B_hat @ X_B.T - torch.outer(v_hat,torch.ones(n_pts)), ord='fro')**2 / n_pts
        #loss += reg * torch.linalg.norm(A_hat, ord='fro')**2 #+ 0.00001 * torch.linalg.norm(B_hat, ord='fro')**2
        loss_val = loss.item()
        loss.backward()
        optimizer.step()
        loss_vals.append(loss_val)
        with torch.no_grad():
            U,S,V = torch.linalg.svd(B_hat)
            S = l1_proj(S, max_on=reg)
            B_hat.data = U @ torch.diag(S) @ V
        # scheduler.step()
    if n_iters > 2:
        plt.plot(loss_vals)
        plt.title("nuc estimation loss")
        plt.show()
    return A_hat.detach().cpu().numpy(), B_hat.detach().cpu().numpy(), v_hat.detach().cpu().numpy()



def estimate_gd_nuc_project_AB_offset_sigmoid(X,Y,reg=0.00001,lr=0.0001,n_iters=1000,transfer0=None,plt_save=None,nuc_reg=False):
    Y = torch.tensor(Y, device='cuda:0').float()
    n_pts,d = Y.shape
    X = torch.tensor(X, device='cuda:0').float()
    X_B = X[:,0:d]
    X_A = X[:,d:]

    if transfer0 is None:
        A_hat = 0.1*np.random.randn(d,d)
        B_hat = 0.1*np.random.randn(d,d)
        v_hat = 0.1*np.random.randn(d)
        scale = 1
    else:
        A_hat = transfer0[0]
        B_hat = transfer0[1]
        v_hat = transfer0[2]
        scale = transfer0[3]
    A_hat = torch.tensor(A_hat, device='cuda:0').float()
    B_hat = torch.tensor(B_hat, device='cuda:0').float()
    v_hat = torch.tensor(v_hat, device='cuda:0').float()
    scale = torch.tensor(1, device='cuda:0').float()
    if nuc_reg:
        U,S,V = torch.linalg.svd(B_hat)
        S = l1_proj(S, max_on=reg)
        B_hat.data = U @ torch.diag(S) @ V
    A_hat.requires_grad_()
    B_hat.requires_grad_()
    v_hat.requires_grad_()
    scale.requires_grad_()
    
    optimizer = optim.SGD([A_hat,B_hat,v_hat], lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    loss_vals = []
    #reg = 0.00001
    for n in range(n_iters):
        optimizer.zero_grad()
        _,S,_ = torch.linalg.svd(B_hat)
        loss = torch.linalg.norm(Y.T - scale * torch.sigmoid(A_hat @ X_A.T + B_hat @ X_B.T- torch.outer(v_hat,torch.ones(n_pts))), ord='fro')**2 / n_pts
        #loss += reg * torch.linalg.norm(A_hat, ord='fro')**2 #+ 0.00001 * torch.linalg.norm(B_hat, ord='fro')**2
        loss_val = loss.item()
        loss.backward()
        optimizer.step()
        loss_vals.append(loss_val)
        if nuc_reg:
            with torch.no_grad():
                U,S,V = torch.linalg.svd(B_hat)
                S = l1_proj(S, max_on=reg)
                B_hat.data = U @ torch.diag(S) @ V
        # scheduler.step()
    if n_iters > 2:
        plt.plot(loss_vals[5:])
        plt.title("sigmoid estimation loss")
        plt.show()
    return A_hat.detach().cpu().numpy(), B_hat.detach().cpu().numpy(), v_hat.detach().cpu().numpy(), scale.detach().cpu().numpy()



def lr_diag_decomp(A_gt, reg):
    # reg = 2000
    A_norm = np.linalg.norm(A_gt, 'fro')**2
    A_gt = torch.tensor(A_gt).float()
    d = A_gt.shape[0]
    D_opt = torch.randn(d, requires_grad=True)
    A_opt = 0.01*np.random.randn(d,d)
    A_opt = torch.tensor(A_opt, device='cuda:0').float()
    A_opt.requires_grad_()
    #A_opt = 0.01*torch.randn(d,d, requires_grad=True)
    optimizer = optim.Adam([D_opt,A_opt], lr=0.1)

    loss_vals = []
    reg = 0.01
    for t in range(500):
        optimizer.zero_grad()
        loss = torch.linalg.norm(A_gt - torch.diag(D_opt) - A_opt, 'fro')**2    
        loss.backward()
        optimizer.step()
        loss_vals.append(loss.item())
        with torch.no_grad():
            U,S,V = torch.linalg.svd(A_opt)
            S = l1_proj(S, max_on=300)
            A_opt.data = U @ torch.diag(S) @ V
        if np.mod(t,10) == 0:
            print(t,loss.item()/A_norm)
    plt.plot(loss_vals[10:])
    plt.show()



def estimate_gd_nuc_project_diag(X,Y,reg=0.00001,lr=None,n_iters=1000,transfer0=None,plt_save=None):
    #lr_diag_decomp(transfer0, reg)
    data = CustomDataset(X,Y)
    batch_size = 512
    X = torch.tensor(X, device='cuda:0').float()
    _,d = X.shape
    Y = torch.tensor(Y, device='cuda:0').float()
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, drop_last=len(X)>batch_size, generator=torch.Generator(device='cuda'))

    if transfer0 is None:
        #u_hat2 = estimate_ls(X,Y)
        A = 0.1*np.random.randn(d,d)
        D = 0.1*np.random.randn(d)
    else:
        A = transfer0 - np.diag(np.diag(transfer0))
        D = np.diag(np.diag(transfer0))
    A = torch.tensor(A, device='cuda:0').float()
    D = torch.tensor(D, device='cuda:0').float()
    U,S,V = torch.linalg.svd(A)
    S = l1_proj(S, max_on=reg)
    A.data = U @ torch.diag(S) @ V
    A.requires_grad_()
    #D.requires_grad_()
    if lr is None:
        optimizer = optim.SGD([A], lr=0.05)
    else:
        optimizer = optim.SGD([A], lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    loss_vals = []
    for n in range(n_iters):
        XY = torch.multiply(Y.T - A @ X.T, X.T).sum(dim=1)
        X2 = torch.multiply(X.T, X.T).sum(dim=1)
        D = torch.divide(XY,X2)
        #print(X.shape,Y.shape)

        #for x_batch, y_batch in data_loader:
        optimizer.zero_grad()
        #loss = torch.linalg.norm(y_batch.T - A @ x_batch.T, ord='fro')**2
        loss = torch.linalg.norm(Y.T - (A + torch.diag(D)) @ X.T, ord='fro')**2
        # loss += 0.001 * torch.linalg.norm(D, 2)**2
        loss_val = loss.item()
        loss.backward()
        optimizer.step()
        loss_vals.append(loss_val)
        with torch.no_grad():
            U,S,V = torch.linalg.svd(A)
            S = l1_proj(S, max_on=reg)
            A.data = U @ torch.diag(S) @ V
        #scheduler.step()
        #print(scheduler.get_last_lr())
        # if np.mod(n,10) == 0:
        #     print(n,loss_val)
        #     print(n,loss_val,torch.linalg.norm(A, 'nuc'))
    plt.plot(loss_vals)
    plt.title('nuc estimation loss diag')
    plt.show()
    if plt_save is not None:
        plt.savefig(plt_save + ".png")
        plt.close()
    return (A + torch.diag(D)).detach().cpu().numpy(), torch.diag(D).detach().cpu().numpy(), A.detach().cpu().numpy()



def estimate_gd_lowrank(X,Y,n_iters=None,rank=5,lr=None,step_multiplier=None,transfer0=None):
    data = CustomDataset(X,Y)
    batch_size = 512
    X = torch.tensor(X, device='cuda:0').float()
    Y = torch.tensor(Y, device='cuda:0').float()
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, drop_last=len(X)>batch_size, generator=torch.Generator(device='cuda'))
    
    if transfer0 is None:
        U = torch.randn(d,rank, device='cuda:0').float()
        V = torch.randn(d,rank, device='cuda:0').float()
        U.requires_grad_()
        V.requires_grad_()
    else:
        U0,S0,V0 = la.svd(transfer0)
        U = torch.tensor(U0[:,0:rank] @ np.diag(np.sqrt(S0[0:rank])), device='cuda:0').float()
        V = torch.tensor(V0[:,0:rank] @ np.diag(np.sqrt(S0[0:rank])), device='cuda:0').float()
        U.requires_grad_()
        V.requires_grad_()
    if lr is None:
        optimizer = optim.Adam([U,V], lr=0.025)
    else:
        optimizer = optim.Adam([U,V], lr=lr)
    loss_vals = []
    total_steps = 1000000
    if step_multiplier is None:
        n_max = np.max([int(total_steps / len(X)), 10])
    else:
        n_max = np.max([int(step_multiplier * total_steps / len(X)), 10])
    n_max = int(step_multiplier * 5000)
    reg = 0.001
    min_loss = None
    best_V = None
    best_U = None
    for n in range(500):
        epoch_loss = 0
#         loss = torch.linalg.norm(Y.T - U @ V.T @ X.T, ord='fro')**2 / len(X)
#         loss += reg * torch.linalg.norm(U, ord='fro')**2 + reg * torch.linalg.norm(V, ord='fro')**2
#         loss_val = loss.item()
#         loss.backward()
#         optimizer.step()
#         loss_vals.append(loss_val)
#         epoch_loss += loss_val
        for x_batch, y_batch in data_loader:
            optimizer.zero_grad()
            loss = torch.linalg.norm(y_batch.T - U @ V.T @ x_batch.T, ord='fro')**2 / batch_size
            loss_reg = reg * torch.linalg.norm(U, ord='fro')**2 + reg * torch.linalg.norm(V, ord='fro')**2
            loss += loss_reg
            loss_val = loss.item()
            loss.backward()
            optimizer.step()
            loss_vals.append(loss_val)
            epoch_loss += loss_val
        if min_loss is None:
            min_loss = epoch_loss
            best_connect = U.clone().detach().cpu().numpy() @ V.clone().detach().cpu().numpy().T
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            best_connect = U.clone().detach().cpu().numpy() @ V.clone().detach().cpu().numpy().T
#         if np.mod(n,10) == 0:
#             print(n,epoch_loss)
    if n_iters is None:
        plt.plot(loss_vals[5:])
        plt.show()
    #connect = U @ V.T
    return best_connect

def estimate_gd_lowrank_alternate(X,Y,n_iters=None,rank=5,lr=None,step_multiplier=None,transfer0=None):
    data = CustomDataset(X,Y)
    batch_size = 512
    X = torch.tensor(X, device='cuda:0').float()
    Y = torch.tensor(Y, device='cuda:0').float()
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, drop_last=len(X)>batch_size, generator=torch.Generator(device='cuda'))
    
    if transfer0 is None:
        U = torch.randn(d,rank, device='cuda:0').float()
        V = torch.randn(d,rank, device='cuda:0').float()
        U.requires_grad_()
        V.requires_grad_()
    else:
        U0,S0,V0 = la.svd(transfer0)
        U = torch.tensor(U0[:,0:rank] @ np.diag(np.sqrt(S0[0:rank])), device='cuda:0').float()
        V = torch.tensor(V0[:,0:rank] @ np.diag(np.sqrt(S0[0:rank])), device='cuda:0').float()
        U.requires_grad_()
        V.requires_grad_()
    if lr is None:
        optimizer_U = optim.Adam([U], lr=0.025)
        optimizer_V = optim.Adam([V], lr=0.025)
    else:
        optimizer_U = optim.Adam([U], lr=lr)
        optimizer_V = optim.Adam([V], lr=lr)
    loss_vals = []
    total_steps = 1000000
    if step_multiplier is None:
        n_max = np.max([int(total_steps / len(X)), 10])
    else:
        n_max = np.max([int(step_multiplier * total_steps / len(X)), 10])
    n_max = int(step_multiplier * 5000)
    reg = 0.001
    min_loss = None
    best_V = None
    best_U = None
    for n in range(200):
        epoch_loss = 0
        for j in range(10):
            for x_batch, y_batch in data_loader:
                optimizer_U.zero_grad()
                loss = torch.linalg.norm(y_batch.T - U @ V.T @ x_batch.T, ord='fro')**2 / batch_size
                loss += reg * torch.linalg.norm(U, ord='fro')**2 + reg * torch.linalg.norm(V, ord='fro')**2
                loss_val = loss.item()
                loss.backward()
                optimizer_U.step()
                loss_vals.append(loss_val)
                epoch_loss += loss_val
        for j in range(10):
            for x_batch, y_batch in data_loader:
                optimizer_V.zero_grad()
                loss = torch.linalg.norm(y_batch.T - U @ V.T @ x_batch.T, ord='fro')**2 / batch_size
                loss += reg * torch.linalg.norm(U, ord='fro')**2 + reg * torch.linalg.norm(V, ord='fro')**2
                loss_val = loss.item()
                loss.backward()
                optimizer_V.step()
                loss_vals.append(loss_val)
                epoch_loss += loss_val

        if min_loss is None:
            min_loss = epoch_loss
            best_connect = U.clone().detach().cpu().numpy() @ V.clone().detach().cpu().numpy().T
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            best_connect = U.clone().detach().cpu().numpy() @ V.clone().detach().cpu().numpy().T
        if np.mod(n,10) == 0:
            print(n,epoch_loss)
    if n_iters is None:
        plt.plot(loss_vals[5:])
        plt.show()
    connect = U @ V.T
    return connect.clone().detach().cpu().numpy()


def estimate_gd_lowrank_alternate2(X,Y,n_iters=None,rank=5,lr=None,step_multiplier=None,transfer0=None):
    data = CustomDataset(X,Y)
    batch_size = 512
    X = torch.tensor(X, device='cuda:0').float()
    Y = torch.tensor(Y, device='cuda:0').float()
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, drop_last=len(X)>batch_size, generator=torch.Generator(device='cuda'))
    d = X.shape[1]
    data_size = X.shape[0]

    if transfer0 is None:
        U = torch.randn(d,rank, device='cuda:0').float()
        V = torch.randn(d,rank, device='cuda:0').float()
        V.requires_grad_()
    else:
        U0,S0,V0 = la.svd(transfer0)
        U = torch.tensor(U0[:,0:rank] @ np.diag(np.sqrt(S0[0:rank])), device='cuda:0').float()
        V = torch.tensor(V0[:,0:rank] @ np.diag(np.sqrt(S0[0:rank])), device='cuda:0').float()
        # U = torch.tensor(U0[:,0:rank], device='cuda:0').float()
        # V = torch.tensor(V0[:,0:rank] @ np.diag(S0[0:rank]), device='cuda:0').float()
        V.requires_grad_()
    if lr is None:
        #optimizer_U = optim.Adam([U], lr=0.025)
        optimizer_V = optim.Adam([V], lr=0.025)
    else:
        #optimizer_U = optim.Adam([U], lr=lr)
        optimizer_V = optim.Adam([V], lr=lr)
    loss_vals = []
    total_steps = 1000000
    # if step_multiplier is None:
    #     n_max = np.max([int(total_steps / len(X)), 10])
    # else:
    #     n_max = np.max([int(step_multiplier * total_steps / len(X)), 10])
    # n_max = int(step_multiplier * 5000)
    reg = 0.001
    min_loss = None
    best_V = None
    best_U = None


    for n in range(500): # 500
        epoch_loss = 0

        with torch.no_grad():
            Xtil = V.T @ X.T
            U = torch.linalg.pinv(Xtil @ Xtil.T) @ Xtil @ Y
            U = U.T
            # U = torch.mean(Y, axis=0)
            # U = U[:,None]
            #U = U / torch.linalg.norm(U)
            
        # for j in range(10):
        #     for x_batch, y_batch in data_loader:
        #         optimizer_U.zero_grad()
        #         loss = torch.linalg.norm(y_batch.T - U @ V.T @ x_batch.T, ord='fro')**2 / batch_size
        #         loss += reg * torch.linalg.norm(U, ord='fro')**2 + reg * torch.linalg.norm(V, ord='fro')**2
        #         loss_val = loss.item()
        #         loss.backward()
        #         optimizer_U.step()
        #         loss_vals.append(loss_val)
        #         epoch_loss += loss_val
        # loss_n = []
        #for j in range(50):
        #     for x_batch, y_batch in data_loader:
        optimizer_V.zero_grad()
        #loss = torch.linalg.norm(y_batch.T - U @ V.T @ x_batch.T, ord='fro')**2 / batch_size
        loss = torch.linalg.norm(Y.T - U @ V.T @ X.T, ord='fro')**2 / data_size
        #loss += reg * torch.linalg.norm(V, ord='fro')**2
        loss_val = loss.item()
        loss.backward()
        optimizer_V.step()
        loss_vals.append(loss_val)
        #loss_n.append(loss_val)
        epoch_loss += loss_val
        # plt.plot(loss_n)
        # plt.show()

        # if min_loss is None:
        #     min_loss = epoch_loss
        #     best_connect = U.clone().detach().cpu().numpy() @ V.clone().detach().cpu().numpy().T
        # if epoch_loss < min_loss:
        #     min_loss = epoch_loss
        #     best_connect = U.clone().detach().cpu().numpy() @ V.clone().detach().cpu().numpy().T
        #if np.mod(n,500) == 0:
        #    print(n,epoch_loss)
    #if n_iters is None:
    plt.plot(loss_vals[5:])
    plt.show()
    #connect = torch.outer(U.flatten(),V.flatten())
    connect = U @ V.T
    return connect.clone().detach().cpu().numpy()


def estimate_gd_lowrank_alternate2_diag(X,Y,n_iters=500,rank=5,lr=None,step_multiplier=None,transfer0=None):
    X = torch.tensor(X, device='cuda:0').float()
    Y = torch.tensor(Y, device='cuda:0').float()
    d = X.shape[1]
    data_size = X.shape[0]

    if transfer0 is None:
        U = torch.randn(d,rank, device='cuda:0').float()
        V = torch.randn(d,rank, device='cuda:0').float()
        D = torch.randn(d, device='cuda:0').float()
        V.requires_grad_()
        D.requires_grad_()
    else:
        U0,S0,V0 = la.svd(transfer0 - np.diag(np.diag(transfer0)))
        U = torch.tensor(U0[:,0:rank] @ np.diag(np.sqrt(S0[0:rank])), device='cuda:0').float()
        V = torch.tensor(V0[:,0:rank] @ np.diag(np.sqrt(S0[0:rank])), device='cuda:0').float()
        D = torch.tensor(np.diag(transfer0), device='cuda:0').float()
        V.requires_grad_()
        D.requires_grad_()
    if lr is None:
        optimizer_V = optim.Adam([V, D], lr=0.025)
    else:
        optimizer_V = optim.Adam([V, D], lr=lr)
    loss_vals = []
    reg = 0.001
    for n in range(n_iters): # 500
        epoch_loss = 0

        with torch.no_grad():
            Xtil = V.T @ X.T
            U = torch.linalg.pinv(Xtil @ Xtil.T) @ Xtil @ (Y - (torch.diag(D) @ X.T).T)
            U = U.T
            # U = torch.mean(Y, axis=0)
            # U = U[:,None]
            #U = U / torch.linalg.norm(U)
            
        optimizer_V.zero_grad()
        #loss = torch.linalg.norm(y_batch.T - U @ V.T @ x_batch.T, ord='fro')**2 / batch_size
        loss = torch.linalg.norm(Y.T - (torch.diag(D) + U @ V.T) @ X.T, ord='fro')**2 / data_size
        #loss += reg * torch.linalg.norm(V, ord='fro')**2 + reg * torch.linalg.norm(D, ord=2)**2
        loss_val = loss.item()
        loss.backward()
        optimizer_V.step()
        loss_vals.append(loss_val)
        epoch_loss += loss_val
        #print(n, loss_val)
    plt.plot(loss_vals[5:])
    plt.show()
    connect = torch.diag(D) + U @ V.T
    return connect.clone().detach().cpu().numpy()


def estimate_gd_lowrank_AB(X,Y,rank=10,lr=0.0001,n_iters=1000,transfer0=None):
    Y = torch.tensor(Y, device='cuda:0').float()
    n_pts,d = Y.shape
    X = torch.tensor(X, device='cuda:0').float()
    X_B = X[:,0:d]
    X_A = X[:,d:]

    if transfer0 is None:
        U = torch.randn(d,rank, device='cuda:0').float()
        V = torch.randn(d,rank, device='cuda:0').float()
        D = torch.randn(d,d, device='cuda:0').float()
        V.requires_grad_()
        D.requires_grad_()
    else:
        U0,S0,V0 = la.svd(transfer0[1])
        U = torch.tensor(U0[:,0:rank] @ np.diag(np.sqrt(S0[0:rank])), device='cuda:0').float()
        V = torch.tensor(V0[:,0:rank] @ np.diag(np.sqrt(S0[0:rank])), device='cuda:0').float()
        D = torch.tensor(transfer0[0], device='cuda:0').float()
        V.requires_grad_()
        U.requires_grad_()
        D.requires_grad_()
    optimizer_V = optim.Adam([V, U, D], lr=lr)
    loss_vals = []
    reg = 0.0001
    for n in range(n_iters): 
        epoch_loss = 0

        # with torch.no_grad():
        #     Xtil = V.T @ X_B.T
        #     U = torch.linalg.pinv(Xtil @ Xtil.T) @ Xtil @ (Y.T - D @ X_A.T).T
        #     U = U.T

        optimizer_V.zero_grad()
        loss = torch.linalg.norm(Y.T - D @ X_A.T - U @ V.T @ X_B.T, ord='fro')**2 / n_pts
        # loss += reg * torch.linalg.norm(V, ord='fro')**2 + reg * torch.linalg.norm(U, ord='fro')**2 + reg * torch.linalg.norm(D, ord=2)**2
        loss_val = loss.item()
        loss.backward()
        optimizer_V.step()
        loss_vals.append(loss_val)
    plt.plot(loss_vals[5:])
    plt.show()
    B = U @ V.T
    return D.clone().detach().cpu().numpy(), B.clone().detach().cpu().numpy()



def estimate_gd_lowrank_AB_offset(X,Y,rank=10,lr=0.0001,n_iters=1000,transfer0=None):
    Y = torch.tensor(Y, device='cuda:0').float()
    n_pts,d = Y.shape
    X = torch.tensor(X, device='cuda:0').float()
    X_B = X[:,0:d]
    X_A = X[:,d:]

    if transfer0 is None:
        U = torch.randn(d,rank, device='cuda:0').float()
        V = torch.randn(d,rank, device='cuda:0').float()
        D = torch.randn(d,d, device='cuda:0').float()
        offset = torch.randn(d, device='cuda:0').float()
        V.requires_grad_()
        U.requires_grad_()
        D.requires_grad_()
        offset.requires_grad_()
    else:
        U0,S0,V0 = la.svd(transfer0[1])
        U = torch.tensor(U0[:,0:rank] @ np.diag(np.sqrt(S0[0:rank])), device='cuda:0').float()
        #V = torch.tensor(V0[:,0:rank] @ np.diag(np.sqrt(S0[0:rank])), device='cuda:0').float()
        V = torch.tensor(np.diag(np.sqrt(S0[0:rank])) @ V0[0:rank,:], device='cuda:0').float()
        D = torch.tensor(transfer0[0], device='cuda:0').float()
        offset = torch.tensor(transfer0[2], device='cuda:0').float()
        V.requires_grad_()
        U.requires_grad_()
        D.requires_grad_()
        offset.requires_grad_()
    optimizer_V = optim.Adam([V, U, D, offset], lr=lr)
    loss_vals = []
    reg = 0.0001
    for n in range(n_iters): 
        epoch_loss = 0

        # with torch.no_grad():
        #     Xtil = V.T @ X_B.T
        #     U = torch.linalg.pinv(Xtil @ Xtil.T) @ Xtil @ (Y.T - D @ X_A.T).T
        #     U = U.T

        optimizer_V.zero_grad()
        loss = torch.linalg.norm(Y.T - D @ X_A.T - U @ V @ X_B.T - torch.outer(offset, torch.ones(n_pts)), ord='fro')**2 / n_pts
        loss += reg * torch.linalg.norm(V, ord='fro')**2 + reg * torch.linalg.norm(U, ord='fro')**2 + reg * torch.linalg.norm(D, ord=2)**2
        loss_val = loss.item()
        loss.backward()
        optimizer_V.step()
        loss_vals.append(loss_val)
    plt.plot(loss_vals[5:])
    plt.show()
    B = U @ V
    return D.clone().detach().cpu().numpy(), B.clone().detach().cpu().numpy(), offset.clone().detach().cpu().numpy()



def estimate_gd_lowrank_alternate_oracle(X,Y,Vst,n_iters=None,rank=5,lr=None,step_multiplier=None,transfer0=None):
    X = torch.tensor(X).float()
    Y = torch.tensor(Y).float()
    Xtil = Vst.T @ X.T
    U = 1 / (Xtil @ Xtil.T) * Xtil @ Y
    U = U.T
    connect = torch.outer(U,Vst)
    return connect.clone().detach().cpu().numpy()


def estimate_gd_l1(X,Y,n_iters=None,reg=0.00001):
    data = CustomDataset(X,Y)
    batch_size = 512
    TX = len(X)
    X = torch.tensor(X, device='cuda:0').float()
    Y = torch.tensor(Y, device='cuda:0').float()
    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, drop_last=len(X)>batch_size)

    #u_hat2 = estimate_ls(X,Y)
    u_hat2 = 0.1*np.random.randn(d,d)
    u_hat = torch.tensor(u_hat2, device='cuda:0').float()
    u_hat.requires_grad_()
    optimizer = optim.SGD([u_hat], lr=0.05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    loss_vals = []
    total_steps = int(400000/5)
    n_max = np.max([int(total_steps / len(X)), 10])
    if n_iters is not None:
        n_max = int(n_iters / len(X))
        if n_max < 1:
            n_max = 1
    #n_max = 100
    for n in range(n_max):
        for x_batch, y_batch in data_loader:
            optimizer.zero_grad()
            loss = torch.linalg.norm(y_batch.T - u_hat @ x_batch.T, ord='fro')**2
            loss += reg*torch.linalg.norm(u_hat, ord=1)
            loss_val = loss.item()
            loss.backward()
            optimizer.step()
            loss_vals.append(loss_val)
        scheduler.step()
        #print(scheduler.get_last_lr())
    if n_iters is None:
        plt.plot(loss_vals)
        plt.show()
    return u_hat.detach().cpu().numpy()

def estimate_mean(X,Y):
    Y = np.array(Y)
    u_hat = np.mean(Y,axis=0)
    return u_hat / la.norm(u_hat)

def estimate_cvxpy_nuc(X,Y,reg=0.000001):
    X = np.array(X)
    Y = np.array(X)
    d = X.shape[1]
    Ahat = cp.Variable((d,d))
    cost = cp.sum_squares(Y.T - Ahat @ X.T) + reg * cp.atoms.norm_nuc.normNuc(Ahat)
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve(verbose=True)
    return Ahat.value



def estimate_gd_lowrank_project(X,Y,rank=25,lr=0.05,n_iters=1000,transfer0=None):
    X = torch.tensor(X, device='cuda:0').float()
    _,d = X.shape
    Y = torch.tensor(Y, device='cuda:0').float()

    if transfer0 is None:
        #u_hat2 = estimate_ls(X,Y)
        u_hat2 = 0.1*np.random.randn(d,d)
    else:
        u_hat2 = transfer0
    u_hat = torch.tensor(u_hat2, device='cuda:0').float()
    U,S,V = torch.linalg.svd(u_hat)
    S[rank:] = 0
    u_hat.data = U @ torch.diag(S) @ V
    u_hat.requires_grad_()
    optimizer = optim.SGD([u_hat], lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    loss_vals = []
    for n in range(n_iters):
        optimizer.zero_grad()
        loss = torch.linalg.norm(Y.T - u_hat @ X.T, ord='fro')**2
        loss_val = loss.item()
        loss.backward()
        optimizer.step()
        loss_vals.append(loss_val)
        with torch.no_grad():
            U,S,V = torch.linalg.svd(u_hat)
            S[rank:] = 0
            u_hat.data = U @ torch.diag(S) @ V
    plt.plot(loss_vals)
    plt.title("projected lowrank estimation loss")
    plt.show()
    return u_hat.detach().cpu().numpy()


def estimate_gd_lowrank_project_diag(X,Y,n_iters=500,rank=5,lr=0.025,step_multiplier=None,transfer0=None):
    X = torch.tensor(X, device='cuda:0').float()
    Y = torch.tensor(Y, device='cuda:0').float()
    d = X.shape[1]
    data_size = X.shape[0]

    if transfer0 is None:
        A = torch.randn(d,d, device='cuda:0').float()
        D = torch.randn(d, device='cuda:0').float()
        A.requires_grad_()
        D.requires_grad_()
    else:
        U0,S0,V0 = la.svd(transfer0)
        S0[rank:] = 0
        A = torch.tensor(U0 @ np.diag(S0) @ V0, device='cuda:0').float()
        D = torch.diag(torch.tensor(transfer0).float() - A)
        A.requires_grad_()
        D.requires_grad_()
    optimizer = optim.SGD([A, D], lr=lr)
    loss_vals = []
    reg = 0.001
    for n in range(n_iters): 
        optimizer.zero_grad()
        loss = torch.linalg.norm(Y.T - (torch.diag(D) + A) @ X.T, ord='fro')**2 / data_size
        #loss += reg * torch.linalg.norm(V, ord='fro')**2 + reg * torch.linalg.norm(D, ord=2)**2
        loss_val = loss.item()
        loss.backward()
        optimizer.step()
        loss_vals.append(loss_val)
        with torch.no_grad():
            U,S,V = torch.linalg.svd(A)
            S[rank:] = 0
            A.data = U @ torch.diag(S) @ V
    plt.plot(loss_vals[5:])
    plt.show()
    connect = torch.diag(D) + A
    return connect.clone().detach().cpu().numpy()

