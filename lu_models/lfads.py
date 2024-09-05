#lfads fluorescence simple mask init state estimate split"
import os
import datetime
import random

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as opt
from torch.autograd import Variable
import matplotlib.pyplot as plt
from utils import batchify_random_sample, update_param_dict

import pdb

#-------------------------
# COST FUNCTION COMPONENTS
#-------------------------
def KLCostGaussian(post_mu, post_lv, prior_mu, prior_lv):
    '''
    KLCostGaussian(post_mu, post_lv, prior_mu, prior_lv)

    KL-Divergence between a prior and posterior diagonal Gaussian distribution.

    Arguments:
        - post_mu (torch.Tensor): mean for the posterior
        - post_lv (torch.Tensor): logvariance for the posterior
        - prior_mu (torch.Tensor): mean for the prior
        - prior_lv (torch.Tensor): logvariance for the prior
    '''
    klc = 0.5 * (prior_lv - post_lv + torch.exp(post_lv - prior_lv) \
         + ((post_mu - prior_mu)/torch.exp(0.5 * prior_lv)).pow(2) - 1.0).mean()
    return klc


def logLikelihoodPoisson(k, lam, mask = None):
    '''
    logLikelihoodPoisson(k, lam, mask)

    Log-likelihood of Poisson distributed counts k given intensity lam.

    Arguments:
        - k (torch.Tensor): Tensor of size batch-size x time-step x input dimensions
        - lam (torch.Tensor): Tensor of size batch-size x time-step x input dimensions
        - mask (torch.Tensor): Optional, Tensor of of size batch-size x time-step x input dimensions
    '''
    if mask is not None:
        loglikelihood = ((k * torch.log(lam) - lam - torch.lgamma(k + 1)) * (1 - mask)).sum()
    else:
        loglikelihood = (k * torch.log(lam) - lam - torch.lgamma(k + 1)).sum()
    return loglikelihood


def logLikelihoodGaussian(x, mu, logvar, mask = None, stim_input = None, stim_weight = 1, untarget_weight = 0.2):
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
        stim_neuron_index, _ = torch.max(stim_input, dim = 1)
        neuron_weights = stim_neuron_index * stim_weight + (1 - stim_neuron_index) * untarget_weight
        loglikelihood = -0.5*((log(2*pi) + logvar + ((x - mu).pow(2)/(torch.exp(logvar)+eps))) * (1 - mask) * neuron_weights).sum()
    else:
        loglikelihood = -0.5*((log(2*pi) + logvar + ((x - mu).pow(2)/(torch.exp(logvar)+eps)))).sum()
    return loglikelihood


def MSE(x, mu, logvar, mask = None, stim_input = None, stim_weight = 1, untarget_weight = 0.2):
    '''
    MSE(x, mu, logvar, mask):
    
    Mean square error of a real-valued observation given a Gaussian distribution with mean 'mu' 
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
        stim_neuron_index, _ = torch.max(stim_input, dim = 1)
        neuron_weights = stim_neuron_index * stim_weight + (1 - stim_neuron_index) * untarget_weight
        mse = -(((x - mu).pow(2)) * (1 - mask) * neuron_weights).sum()
    else:
        mse = -(x - mu).pow(2).sum()
    return mse

def spatial_inputs(distance_slm, distance_stim, slm_constant = 200, stim_constant = 10):
    inputs = torch.exp(-distance_stim/stim_constant) * torch.exp(-distance_slm/slm_constant)
    #inputs = 1/stim_constant * 1/slm_constant * torch.exp(-0.5*torch.square(distance_stim/stim_constant)) * torch.exp(-0.5*torch.square(distance_slm/slm_constant))
    return inputs

#--------
# NETWORK
#--------

class LFADS_Net(nn.Module):
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    def __init__(self, inputs_dim, T, T_init, dt,
                 groups_dim = None,
                 model_hyperparams=None,
                 device = 'cpu', save_variables = False,
                 seed = None,
                 stimPosition = None,
                 slmDist = None, 
                 stimDist = None,
                 Position_X = None,
                 Position_Y = None):
        '''
        LFADS_Net (Latent Factor Analysis via Dynamical Systems) neural network class.
        
        __init__(self, inputs_dim, T, dt,
                 model_hyperparams=None,
                 device='cpu', save_variables=False)
                 
            required arguments:
            
            - inputs_dim (int): the dimensionality of the data (e.g. number of cells)
            - T (int): number of time-steps in one sequence (i.e. one data point)
            - T_init (int): number of time-steps for init state estimation (i.e. one data point)
            - dt (float): time-step in seconds
            
            optional arguments:
            - model_hyperparams (dict) : dictionary of model_hyperparameters
                - ### DATA HYPERPARAMETERS ### 
                - dataset_name (String): name given to identify dataset (default = 'unknown')
                - run_name (String): name given to identify model run (default = 'tmp')
                
                - ### MODEL HYPERPARAMETERS ###
                - g_dim (int): dimensionality of the generator (default = 100)
                - u_dim (int): dimensionality of the inferred inputs to the generator (default = 1)
                - g0_encoder_dim (int): dimensionality of the encoder for the initial conditions for the generator
                                        (default = 100)
                - c_encoder_dim (int): dimensionality of the encoder for the controller (default = 100)
                - controller_dim (int): dimensionality of the controller (default = 100)
                - g0_prior_logkappa (float): initial log-variance for the learnable prior over the initial 
                                             generator state (default = log(0.1))
                - u_prior_logkappa (float): initial log-variance for the leanable prior over the inferred inputs
                                            to generator (default = log(0.1))
                - keep_prob (float): keep probability for drop-out layers, if < 1 (default = 1.0)
                - clip_val (float): clips the hidden unit activity to be less than this value (default = 5.0)
                - max_norm (float): maximum gradient norm (default=200.0)
                
                - ### OPTIMIZER HYPERPARAMETERS ###
                - lr (float): learning rate for ADAM optimizer (default = 0.01)
                - eps (float): epsilon value for ADAM optimizer (default = 0.1)
                - betas (2-tuple of floats): beta values for ADAM optimizer (default = (0.9, 0.999))
                - lr_decay (float): learning rate decay factor (default = 0.95)
                - lr_min (float): minimum learning rate (default = 1e-5)
                - scheduler_on (bool): apply scheduler if True (default = True)
                - scheduler_patience (int): number of steps without loss decrease before weight decay (default = 6)
                - scheduler_cooldown (int): number of steps after weight decay to wait before next weight decay (default = 6)
                - kl_weight_schedule_start (int) : optimisation step to start kl_weight increase (default = 0)
                - kl_weight_schedule_dur (int) : number of optimisation steps to increase kl_weight to 1.0 (default = 2000)
                - l2_weight_schedule_start (int) : optimisation step to start l2_weight increase (default = 0)
                - l2_weight_schedule_dur (int) : number of optimisation steps to increase l2_weight to 1.0 (default = 2000)
                - l2_gen_scale (float) : scaling factor for regularising l2 norm of generator hidden weights  (default = 0.0)
                - l2_con_scale (float) : scaling factor for regularising l2 norm of controller hidden weights (default = 0.0)
            
            - device (String): device to use (default= 'cpu')
            - save_variables (bool) : whether to save dynamic variables (default= False)
        '''
        
        # -----------------------
        # BASIC INIT STUFF
        # -----------------------
        
        # call the nn.Modules constructor
        super(LFADS_Net, self).__init__()
        
        # Default hyperparameters
        default_hyperparams  = {### DATA PARAMETERS ###
                                'dataset_name'             : 'unknown',
                                'run_name'                 : 'tmp',
                                
                                ### MODEL PARAMETERS ### 
                                'g_dim'                    : 100,
                                'u_dim'                    : 1,
                                'g0_encoder_dim'           : 100,
                                'c_encoder_dim'            : 100,
                                'controller_dim'           : 100,
                                'g0_prior_kappa'           : 0.1,
                                'u_prior_kappa'            : 0.1,
                                'keep_prob'                : 1.0,
                                'clip_val'                 : 5.0,
                                'max_norm'                 : 200,
                                'use_controller_feedback'  : True,
                                'learned_init'             : False,
                                'loss_form'                : 'logLikelihoodGaussian',
                                'stim_form'                : 'neuron',
                                'stim_constant_train'      : False,
            
                                ### OPTIMIZER PARAMETERS 
                                'learning_rate'            : 0.01,
                                'learning_rate_min'        : 1e-5,
                                'learning_rate_decay'      : 0.95,
                                'scheduler_on'             : True,
                                'scheduler_patience'       : 6,
                                'scheduler_cooldown'       : 6,
                                'epsilon'                  : 0.1,
                                'betas'                    : (0.9, 0.99),
                                'l2_gen_scale'             : 0.0,
                                'l2_con_scale'             : 0.0,
                                'kl_weight_schedule_start' : 0,
                                'kl_weight_schedule_dur'   : 2000,
                                'kl_weight_scale'          : 1.0,        
                                'l2_weight_schedule_start' : 0,
                                'l2_weight_schedule_dur'   : 2000,
                                'l2_weight_scale'          : 1.0,
                                'untarget_weight'          : 0.2,
                                'ew_weight_schedule_start' : 0,
                                'ew_weight_schedule_dur'   : 2000,
                                'em_weight_scale'          : 1.0}
        
        # Store the hyperparameters        
        self._update_params(default_hyperparams, model_hyperparams)
        
        self.inputs_dim                = inputs_dim
        self.groups_dim                = groups_dim
        self.T                         = T
        self.T_init                    = T_init
        self.dt                        = dt

        self.device                    = device
        self.save_variables            = save_variables
        self.seed                      = seed
        self.stimPosition              = stimPosition
        self.slmDist                   = slmDist
        self.stimDist                  = stimDist
        self.Position_X                = Position_X
        self.Position_Y                = Position_Y
        
        if self.seed is None:
            self.seed = random.randint(1, 10000)
            print('Random seed: {}'.format(self.seed))
        else:
            print('Preset seed: {}'.format(self.seed))
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed_all(self.seed)
        
        # Store loss
        self.full_loss_store = {'train_loss' : {}, 'train_recon_loss' : {}, 'train_kl_loss' : {},
                                'valid_loss' : {}, 'valid_recon_loss' : {}, 'valid_kl_loss' : {},
                                'l2_loss' : {}}
        self.train_loss_store = []
        self.valid_loss_store = []
        self.best = np.inf
        
        # Training variable
        self.epochs = 0
        self.current_step = 0
        self.last_decay_epoch = 0
        self.cost_weights = {'kl' : {'weight': 0, 'schedule_start': self.kl_weight_schedule_start,
                                     'schedule_dur': self.kl_weight_schedule_dur,
                                     'scale': self.kl_weight_scale},
                             'l2' : {'weight': 0, 'schedule_start': self.l2_weight_schedule_start,
                                     'schedule_dur': self.l2_weight_schedule_dur,
                                     'scale': self.l2_weight_scale}}
        
        # -----------------------
        # NETWORK LAYERS INIT
        # 
        # Notation:
        #
        #   layertype_outputvariable(_direction)
        #
        #   Examples: gru_Egen_forward = "gated recurrent unit layer, encoder for generator, forward direction"           
        # -----------------------
        
        # ----
        # RNN layers
        # ----

        # Generator Forward Encoder
        self.gru_Egen_forward  = nn.GRUCell(input_size= self.inputs_dim, hidden_size= self.g0_encoder_dim)
        
        # Generator Backward Encoder
        self.gru_Egen_backward = nn.GRUCell(input_size= self.inputs_dim, hidden_size= self.g0_encoder_dim)
        
        # Controller Forward Encoder
        self.gru_Econ_forward  = nn.GRUCell(input_size= self.inputs_dim, hidden_size= self.c_encoder_dim)
        
        # Controller Backward Encoder
        self.gru_Econ_backward = nn.GRUCell(input_size= self.inputs_dim, hidden_size= self.c_encoder_dim)
        
        # Controller
        if self.use_controller_feedback:
            self.gru_controller = nn.GRUCell(input_size = self.c_encoder_dim * 2 + self.inputs_dim, hidden_size= self.controller_dim)
        else:
            self.gru_controller = nn.GRUCell(input_size = self.c_encoder_dim * 2, hidden_size= self.controller_dim)
        
        # Generator
        if self.stim_form == 'neuron' or self.stim_form == 'zero' or self.stim_form == 'spatial':
            self.gru_generator = nn.GRUCell(input_size= self.inputs_dim, hidden_size= self.g_dim)
            print(self.stim_form)
        elif self.stim_form == 'group':
            print(self.stim_form)
            self.gru_generator = nn.GRUCell(input_size= self.groups_dim, hidden_size= self.g_dim)
        else:
            raise ValueError("Invalid stim_form")
        
        # -----------
        # Fully connected layers
        # -----------
        
        # mean and logvar of the posterior distribution for the generator initial conditions (g0 from E_gen)
        # takes as inputs:
        #  - the forward encoder for g0 at time T (g0_enc_f_T)
        #  - the backward encoder for g0 at time 1 (g0_enc_b_0]
        self.fc_g0mean   = nn.Linear(in_features= 2 * self.g0_encoder_dim, out_features = self.g_dim)
        self.fc_g0logvar = nn.Linear(in_features= 2 * self.g0_encoder_dim, out_features = self.g_dim)
        
        # mean and logvar of the posterior distribution for the inferred inputs (u provided to g)
        # takes as inputs:
        #  - the controller at time t (c_t)
        self.fc_umean   = nn.Linear(in_features= self.controller_dim, out_features= self.u_dim)
        self.fc_ulogvar = nn.Linear(in_features= self.controller_dim, out_features= self.u_dim)
        
        # output mu from generator
        self.fc_mu   = nn.Linear(in_features= self.g_dim, out_features= self.inputs_dim)
        
        # output logvar from generator
        self.fc_logvar   = nn.Linear(in_features= self.g_dim, out_features= self.inputs_dim)
        
        # -----------
        # Dropout layer
        # -----------
        self.dropout = nn.Dropout(1.0 - self.keep_prob)
        
        # -----------------------
        # WEIGHT INIT
        # 
        # The weight initialization is modified from the standard PyTorch, which is uniform. Instead,
        # the weights are drawn from a normal distribution with mean 0 and std = 1/sqrt(K) where K
        # is the size of the input dimension. This helps prevent vanishing/exploding gradients by
        # keeping the eigenvalues of the Jacobian close to 1.
        # -----------------------
        
        # Step through all layers and adjust the weight initiazition method accordingly
        for m in self.modules():
            
            # GRU layer, update using input weight and recurrent weight dimensionality
            if isinstance(m, nn.GRUCell):
                k_ih = m.weight_ih.shape[1] # dimensionality of the inputs to the GRU
                k_hh = m.weight_hh.shape[1] # dimensionality of the GRU outputs
                m.weight_ih.data.normal_(std = k_ih ** -0.5) # inplace resetting of W ~ N(0,1/sqrt(N))
                m.weight_hh.data.normal_(std = k_hh ** -0.5) # inplace resetting of W ~ N(0,1/sqrt(N))
            
            # FC layer, update using input dimensionality
            elif isinstance(m, nn.Linear):
                k = m.in_features # dimensionality of the inputs
                m.weight.data.normal_(std = k ** -0.5) # inplace resetting of W ~ N(0,1/sqrt(N))
        
        # ---------------------------
        # FIXED PRIOR PARAMETERS INIT
        # ---------------------------
        
        self.g0_prior_mu = nn.parameter.Parameter(torch.tensor(0.0), requires_grad = False)
        self.u_prior_mu  = nn.parameter.Parameter(torch.tensor(0.0), requires_grad = False)
        
        # self.slm_constant = nn.parameter.Parameter(torch.tensor(200.0), requires_grad = True)
        # self.stim_constant = nn.parameter.Parameter(torch.tensor(10.0), requires_grad = True)
        
        # ---------------------------
        # Learned Initial State
        # ---------------------------
        
        if self.learned_init:
            self.g0_init = nn.parameter.Parameter(nn.init.normal_(torch.empty(self.g_dim)), requires_grad = True)
        
        from math import log
        self.g0_prior_logkappa = nn.parameter.Parameter(torch.tensor(log(self.g0_prior_kappa)), requires_grad = False)
        self.u_prior_logkappa  = nn.parameter.Parameter(torch.tensor(log(self.u_prior_kappa)), requires_grad = False)
    
        # --------------------------
        # OPTIMIZER INIT
        # --------------------------
        self.optimizer = opt.Adam(self.parameters(), lr=self.learning_rate, eps=self.epsilon, betas=self.betas)
        
        # --------------------------
        # LOG-LIKELIHOOD FUNCTION
        # --------------------------
        
        print('loss_form:', self.loss_form)
        if self.loss_form == 'logLikelihoodGaussian':
            self.logLikelihood = logLikelihoodGaussian
        elif self.loss_form == 'MSE':
            self.logLikelihood = MSE
        
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    def initialize(self, batch_size=None):
        '''
        initialize()
        
        Initialize dynamic model variables. These need to be reinitialized with each forward pass to
        ensure we don't need to retain graph between each .backward() call. 
        
        See https://discuss.pytorch.org/t/what-exactly-does-retain-variables-true-in-loss-backward-do/3508/2
        for discussion and explanation
        
        Note: The T + 1 terms  accommodate learnable biases for all variables, except for the generator,
        which is provided with a g0 estimate from the network
        
        optional arguments:
          batch_size (int) : batch dimension. If None, use self.batch_size.
        
        '''
        
        batch_size = batch_size if batch_size is not None else self.batch_size
        
        self.g0_prior_mean = torch.ones(batch_size, self.g_dim).to(self.device)*self.g0_prior_mu            # g0 prior mean
        self.u_prior_mean  = torch.ones(batch_size, self.u_dim).to(self.device)*self.u_prior_mu             # u prior mean
        
        self.g0_prior_logvar = torch.ones(batch_size, self.g_dim).to(self.device)*self.g0_prior_logkappa    # g0 prior logvar
        self.u_prior_logvar  = torch.ones(batch_size, self.u_dim).to(self.device)*self.u_prior_logkappa     # u prior logvar
        
        self.c = Variable(torch.zeros((batch_size, self.controller_dim)).to(self.device))  # Controller hidden state
        
        self.efgen = Variable(torch.zeros((batch_size, self.g0_encoder_dim)).to(self.device))  # Forward generator encoder
        self.ebgen = Variable(torch.zeros((batch_size, self.g0_encoder_dim)).to(self.device))  # Backward generator encoder
        
        self.efcon = torch.zeros((batch_size, self.T+1, self.c_encoder_dim)).to(self.device)   # Forward controller encoder
        self.ebcon = torch.zeros((batch_size, self.T+1, self.c_encoder_dim)).to(self.device)   # Backward controller encoder
            
        if self.save_variables:
            # self.inputs        = torch.zeros(batch_size, self.T, self.u_dim)
            # self.inputs_mean   = torch.zeros(batch_size, self.T, self.u_dim)
            # self.inputs_logvar = torch.zeros(batch_size, self.T, self.u_dim)
            self.outputs_mean  = torch.zeros(batch_size, self.T, self.inputs_dim)
            self.outputs_logvar = torch.zeros(batch_size, self.T, self.inputs_dim)

    def encode(self, x, sampling):
        '''
        encode(x)
        
        Function to encode the data with the forward and backward encoders.
        
        Arguments:
          - x (torch.Tensor): Variable tensor of size batch size x time-steps x input dimension
        '''
        
        # Dropout some data
        if self.keep_prob < 1.0:
            x = self.dropout(x)
        
        # Encode data into forward and backward generator encoders to produce E_gen
        # for generator initial conditions.
        for t in range(1, self.T_init + 1):
            
            self.efgen = torch.clamp(self.gru_Egen_forward(x[:, t - 1], self.efgen), max=self.clip_val)
            self.ebgen = torch.clamp(self.gru_Egen_backward(x[:, self.T_init - t], self.ebgen), max=self.clip_val)
        
        # Concatenate efgen_T and ebgen_1 for generator initial condition sampling
        egen = torch.cat((self.efgen, self.ebgen), dim=1)

        # Dropout the generator encoder output
        if self.keep_prob < 1.0:
            egen = self.dropout(egen)
            
        # Sample initial conditions for generator from g0 posterior distribution
        self.g0_mean   = self.fc_g0mean(egen)
        self.g0_logvar = torch.clamp(self.fc_g0logvar(egen), min=np.log(0.0001))
        if sampling:
            if self.learned_init:
                self.g = self.g0_init.unsqueeze(0).repeat(self.batch_size, 1)
            else:
                self.g = Variable(torch.randn(self.batch_size, self.g_dim).to(self.device))*torch.exp(0.5*self.g0_logvar)\
                                 + self.g0_mean
        else:
            if self.learned_init:
                self.g = self.g0_init.unsqueeze(0).repeat(self.batch_size, 1)
            else:
                self.g = self.g0_mean

        # KL cost for g(0)
        self.kl_loss   = KLCostGaussian(self.g0_mean, self.g0_logvar,
                                        self.g0_prior_mean, self.g0_prior_logvar)/x.shape[0]
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    def generate(self, x, mask, stim_input, stim_group_input = None, infer_input_mask = None, sample = False):
        '''
        generate()
        
        Generates the rates using the controller encoder outputs and the sampled initial conditions for
        generator.
        '''
        
        self.recon_loss = 0
        
        for t in range(self.T_init, self.T):
            # Combine with known inputs
            if self.stim_form == 'neuron':
                u_input = stim_input[:, t, :]
            elif self.stim_form == 'group':
                u_input = stim_group_input[:, t, :]
            elif self.stim_form == 'zero':
                u_input = torch.zeros_like(stim_input[:, t, :])
            elif self.stim_form == 'spatial':
                distance_slm = torch.mm(self.slmDist, stim_group_input[:, t, :].T).T
                distance_stim = torch.mm(self.stimDist, stim_group_input[:, t, :].T).T
                stim_on, _ = torch.max(stim_input[:, t, :], dim = 1)
                stim_on_time = torch.ones_like(stim_input[:, t, :]) * stim_on[:, None]
                u_input = stim_on_time * spatial_inputs(distance_slm, distance_stim)
            else:
                raise ValueError("Invalid stim_form")
            
            # Update generator
            self.g = torch.clamp(self.gru_generator(u_input, self.g), min=0.0, max=self.clip_val)

            # Dropout on generator output
            if self.keep_prob < 1.0:
                self.g = self.dropout(self.g)
            
            # Generate output Gaussain mu from generator state
            self.mu = self.fc_mu(self.g)
            
            # Generate output Gaussain logvar from generator state
            self.logvar = self.fc_logvar(self.g)
    
            # Reconstruction loss
            if mask is not None:
                self.recon_loss = self.recon_loss - self.logLikelihood(x[:, t], self.mu, self.logvar, mask[:, t], stim_input, self.untarget_weight)/x.shape[0]
            else:
                self.recon_loss = self.recon_loss - self.logLikelihood(x[:, t], self.mu, self.logvar)/x.shape[0]
            
            if self.save_variables:
                # self.inputs[:, t] = self.u.detach().cpu()
                # self.inputs_mean[:, t] = self.u_mean.detach().cpu()
                # self.inputs_logvar[:, t] = self.u_logvar.detach().cpu()
                self.outputs_mean[:, t]   = self.mu.detach().cpu()
                self.outputs_logvar[:, t]   = self.logvar.detach().cpu()     
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    def forward(self, x, mask = None, stim_input = None, stim_group_input = None, infer_input_mask = None, sampling = True):
        '''
        forward(x, mask)
        
        Runs a forward pass through the network.
        
        Arguments:
          - x (torch.Tensor): Single-trial fluorescence data. Tensor of size batch size x time-steps x input dimension
        '''
        
        batch_size, steps_dim, inputs_dim = x.shape
        
        assert steps_dim  == self.T
        assert inputs_dim == self.inputs_dim
        
        self.batch_size = batch_size
        self.initialize(batch_size = batch_size)
        self.encode(x, sampling)
        self.generate(x, mask, stim_input, stim_group_input, infer_input_mask, sampling)
        
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    def reconstruct(self, x, sampling = True):
        '''
        reconstruct(x)
        
        Runs a forward pass through the network, and outputs reconstruction of data x. History is not tracked.
        
        Arguments:
          - x (torch.Tensor): Single-trial fluo data. Tensor of size batch size x time-steps x input dimensions
          
        Returns:
          - rates (torch.Tensor): Reconstructed fluo gaussian distribution mean. Tensor of size batch size x time-steps x input dimensions
        '''
        self.eval()
        f = x[0]
        mask = x[1]
        stim_input = x[2]
        stim_group_input = x[3]
        infer_input_mask = x[4]
        self.batch_size = f.shape[0]
        prev_save = self.save_variables
        with torch.no_grad():
            self.save_variables = True
            self(f, mask, stim_input, stim_group_input, infer_input_mask, sampling)
        
        self.save_variables = prev_save # reset to previous value
        
        return self.outputs_mean.mean(dim=0).cpu().numpy(), self.outputs_logvar.mean(dim=0).cpu().numpy()

    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------

    def weight_schedule_fn(self, step):
        '''
        weight_schedule_fn(step)
        
        Calculate the KL and L2 regularization weights from the current training step number. Imposes
        linearly increasing schedule on regularization weights to prevent early pathological minimization
        of KL divergence and L2 norm before sufficient data reconstruction improvement. See bullet-point
        4 of section 1.9 in online methods
        
        required arguments:
        step (int) : training step number
        '''
        
        for cost_key in self.cost_weights.keys():
            # Get step number of scheduler
            weight_step = max(step - self.cost_weights[cost_key]['schedule_start'], 0)
            
            # Calculate schedule weight
            self.cost_weights[cost_key]['weight'] = self.cost_weights[cost_key]['scale'] * min(weight_step/ self.cost_weights[cost_key]['schedule_dur'], 1.0)
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    def apply_decay(self, current_loss):
        '''
        apply_decay(current_loss)
        
        Decrease the learning rate by a defined factor (self.learning_rate_decay) if loss is greater
        than the loss in the last six training steps and if the loss has not decreased in the last
        six training steps. See bullet point 8 of section 1.9 in online methods
        '''
        
        if len(self.train_loss_store) >= self.scheduler_patience:
            if all((current_loss > past_loss for past_loss in self.train_loss_store[-self.scheduler_patience:])):
                if self.epochs >= self.last_decay_epoch + self.scheduler_cooldown:
                    self.learning_rate  = self.learning_rate * self.learning_rate_decay
                    self.last_decay_epoch = self.epochs
                    for g in self.optimizer.param_groups:
                        g['lr'] = self.learning_rate
                    print('Learning rate decreased to %.8f'%self.learning_rate)
            
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    def test(self, l2_loss, dl=None, dataset=None, batch_size=4):
        self.eval()
        if dl is None:
            if dataset is None:
                raise IOError('Must pass either a dataset or a dataloader.')
            else:
                self.batch_size = batch_size
                dl = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
        elif dataset is not None:
            print('If both a dataloader and a dataset are passed, the\ndataloader is used.')
            
        else:
            self.batch_size = dl.batch_size

        test_loss = 0
        test_recon_loss = 0
        test_kl_loss = 0
        
        # Collect separate weighted losses
        kl_weight = self.cost_weights['kl']['weight']
        l2_weight = self.cost_weights['l2']['weight']
            
        for i, x in enumerate(dl, 0):
            with torch.no_grad():
                f = Variable(x[0])
                mask = Variable(x[1])
                stim_input = Variable(x[2])
                if len(x) > 3:
                    stim_group_input = Variable(x[3])
                else:
                    stim_group_input = None
                if len(x) > 4:
                    infer_input_mask = Variable(x[4])
                else:
                    infer_input_mask = None
                self(f, mask, stim_input, stim_group_input, infer_input_mask)
                loss = self.recon_loss + kl_weight * self.kl_loss + l2_weight * l2_loss
                test_loss += loss.data
                test_recon_loss += self.recon_loss.data
                test_kl_loss += self.kl_loss.data              
                
        test_loss /= (i+1)
        test_recon_loss /= (i+1)
        test_kl_loss /= (i+1)
        return test_loss, test_recon_loss, test_kl_loss, 
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    def fit(self, train_dataset, valid_dataset,
            batch_size=4, max_epochs=100,
            use_tensorboard=False, health_check=False,
            train_truth=None, valid_truth=None, output='.'):
        '''
        fit(self, train_dataset, valid_dataset, train_params=None, train_truth=None, valid_truth=None)        
        Fits the LFADS_Net using ADAM optimization.
        
        required arguments:
            - train_dataset (torch.utils.data.TensorDataset): Dataset with the training data to fit LFADS model
            - valid_dataset (torch.utils.data.TensorDataset): Dataset with validation data to validate LFADS model
        
        optional arguments:
            - batch_size (int) : number of data points in batch (default = 4)
            - max_epochs (int) : number of epochs to run in loop (default = 100)
            - use_tensorboard (bool) : whether to write results to tensorboard (default = False)
            - health_check (bool)    : whether to calculate weight and gradient norms
            - train_truth (torch.Tensor) : ground-truth rates for training dataset
            - valid_truth (torch.Tensor) : ground-truth rates for validation dataset
        '''
        # Set Training Loop parameters
        self.batch_size = batch_size

        # create the dataloader
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size)
        
        # Initialize directory to save checkpoints
        save_loc = '%s/models/%s/%s/checkpoints/'%(output, self.dataset_name, self.run_name)

        # Create model_checkpoint directory if it doesn't exist
        if not os.path.isdir(save_loc):
            os.makedirs(save_loc)
        elif os.path.exists(save_loc) and self.epochs==0:
            os.system('rm -rf %s'%save_loc)
            os.makedirs(save_loc)
            
        # Initialize tensorboard
        if use_tensorboard:
            tb_folder = '%s/models/%s/%s/tensorboard/'%(output, self.dataset_name, self.run_name)
            if not os.path.exists(tb_folder):
                os.mkdir(tb_folder)
            elif os.path.exists(tb_folder) and self.epochs==0:
                os.system('rm -rf %s'%tb_folder)
                os.mkdir(tb_folder)
            
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(tb_folder)
            
        # print a message
        print('Beginning training...')

        # for each epoch...
        for epoch in range(max_epochs):
            self.train()
            # If minimum learning rate reached, break training loop
            if self.learning_rate <= self.learning_rate_min:
                break
            
            # cumulative training loss for this epoch
            train_loss = 0
            train_recon_loss = 0
            train_kl_loss = 0
            train_l2 = 0
            
            # for each batch...
            for i, x in enumerate(train_dl, 0):
                self.current_step += 1
                                                
                # apply Variable wrapper to batch
                f = Variable(x[0])
                
                mask = Variable(x[1])
                
                stim_input = Variable(x[2])
                
                if len(x) > 3:
                    stim_group_input = Variable(x[3])
                else:
                    stim_group_input = None
                if len(x) > 4:
                    infer_input_mask = Variable(x[4])
                else:
                    infer_input_mask = None

                # zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Calculate regularizer weights
                self.weight_schedule_fn(self.current_step)
                
                # Forward
                self(f, mask, stim_input, stim_group_input, infer_input_mask)
                
                # Calculate l2 regularisation penalty
                l2_loss = self.l2_gen_scale * self.gru_generator.weight_hh.norm(2)/self.gru_generator.weight_hh.numel() + \
                          self.l2_con_scale * self.gru_controller.weight_hh.norm(2)/self.gru_controller.weight_hh.numel()
                
                # Collect separate weighted losses
                kl_weight = self.cost_weights['kl']['weight']
                l2_weight = self.cost_weights['l2']['weight']
                
                loss = self.recon_loss + kl_weight * self.kl_loss + l2_weight * l2_loss
                
                # Check if loss is nan
                assert not torch.isnan(loss.data), 'Loss is NaN'
                                
                # Backward
                loss.backward()
                
                # clip gradient norm
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_norm)
                
                # update the weights
                self.optimizer.step()
                
                if use_tensorboard:
                    self.health_check(writer)
                
                
                # Add batch loss to epoch running loss
                train_loss += loss.data
                train_recon_loss += self.recon_loss.data
                train_kl_loss += self.kl_loss.data    
                train_l2 += l2_loss        
            
            train_loss /= (i+1)
            train_recon_loss /= (i+1)
            train_kl_loss /= (i+1)
            train_l2 /= (i+1)
            
            valid_loss, valid_recon_loss, valid_kl_loss = self.test(l2_loss, dl=valid_dl)
                
            # Print Epoch Loss
            print('Epoch: %4d, Step: %5d, training loss: %.3f, train_recon_loss: %.3f, train_kl: %.3f, train_l2: %.3f' %(self.epochs+1, self.current_step, train_loss, train_recon_loss, train_kl_loss, train_l2))
            print('Epoch: %4d, Step: %5d, validation loss: %.3f, valid_recon_loss: %.3f, valid_kl: %.3f' %(self.epochs+1, self.current_step, valid_loss, valid_recon_loss, valid_kl_loss))
            # Apply learning rate decay function
            if self.scheduler_on:
                self.apply_decay(train_loss)
                
            # Store loss
            self.train_loss_store.append(float(train_loss))
            self.valid_loss_store.append(float(valid_loss))
            
            self.full_loss_store['train_loss'][self.epochs]       = float(train_loss)
            self.full_loss_store['train_recon_loss'][self.epochs] = float(train_recon_loss)
            self.full_loss_store['train_kl_loss'][self.epochs]    = float(train_kl_loss)
            self.full_loss_store['valid_loss'][self.epochs]       = float(valid_loss)
            self.full_loss_store['valid_recon_loss'][self.epochs] = float(valid_recon_loss)
            self.full_loss_store['valid_kl_loss'][self.epochs]    = float(valid_kl_loss)
            self.full_loss_store['l2_loss'][self.epochs]          = float(l2_loss.data)
            
            # Write results to tensorboard
            if use_tensorboard:
                
                writer.add_scalars('1_Loss/1_Total_Loss', {'Training' : float(train_loss), 
                                                         'Validation' : float(valid_loss)}, self.epochs)

                writer.add_scalars('1_Loss/2_Reconstruction_Loss', {'Training' :  float(train_recon_loss), 
                                                                  'Validation' : float(valid_recon_loss)}, self.epochs)
                
                writer.add_scalars('1_Loss/3_KL_Loss' , {'Training' : float(train_kl_loss), 
                                                       'Validation' : float(valid_kl_loss)}, self.epochs)
                
                writer.add_scalar('1_Loss/4_L2_loss', float(l2_loss.data), self.epochs)
                
                writer.add_scalar('2_Optimizer/1_Learning_Rate', self.learning_rate, self.epochs)
                writer.add_scalar('2_Optimizer/2_KL_weight', kl_weight, self.epochs)
                writer.add_scalar('2_Optimizer/3_L2_weight', l2_weight, self.epochs)

            self.epochs += 1
            
            if self.valid_loss_store[-1] < self.best:
                self.last_saved = epoch
                self.best = self.valid_loss_store[-1]
                # saving checkpoint
                self.save_checkpoint(output=output)

                if use_tensorboard:
                    figs_dict_train = self.plot_summary(data= train_dataset, truth= train_truth)
                    writer.add_figure('Examples/1_Train', figs_dict_train['traces'], self.epochs, close=True)
                    writer.add_figure('Inputs/1_Train', figs_dict_train['inputs'], self.epochs, close=True)

                    figs_dict_valid = self.plot_summary(data= valid_dataset, truth= valid_truth)
                    writer.add_figure('Examples/2_Valid', figs_dict_valid['traces'], self.epochs, close=True)
                    writer.add_figure('Inputs/2_Valid', figs_dict_valid['inputs'], self.epochs, close=True)

                    if train_truth is not None:
                        writer.add_figure('Ground_truth/1_Train', figs_dict_train['truth'], self.epochs, close=True)
                        writer.add_figure('R-squared/1_Train', figs_dict_train['rsq'], self.epochs, close=True)

                    if valid_truth is not None:
                        writer.add_figure('Ground_truth/2_Valid', figs_dict_valid['truth'], self.epochs, close=True)
                        writer.add_figure('R-squared/2_Valid', figs_dict_valid['rsq'], self.epochs, close=True)
                        
        if use_tensorboard:
            writer.close()
        
        import pandas as pd
        df = pd.DataFrame(self.full_loss_store)
        df.to_csv('%s/models/%s/%s/loss.csv'%(output, self.dataset_name, self.run_name), index_label='epoch')
        
        # Save a final checkpoint
        self.save_checkpoint(force=True, output=output, name = 'final')
        
        # Print message
        print('...training complete.')
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    def plot_traces(self, pred, true,figsize=(8,8), num_traces=12, ncols=2, mode=None, norm=True, pred_logvar=None):
        '''
        Plot fitted intensity function and compare to ground truth
        
        Arguments:
            - pred (np.array): array of predicted values to plot (dims: num_steps x num_cells)
            - true (np.array)   : array of true values to plot (dims: num_steps x num_cells)
            - figsize (2-tuple) : figure size (width, height) in inches (default = (8, 8))
            - num_traces (int)  : number of traces to plot (default = 24)
            - ncols (int)       : number of columns in figure (default = 2)
            - mode (string)     : mode to select subset of traces. Options: 'activity', 'rand', None.
                                  'Activity' plots the the num_traces/2 most active traces and num_traces/2
                                  least active traces defined sorted by mean value in trace
            - norm (bool)       : normalize predicted and actual values (default=True)
            - pred_logvar (np.array) : array of predicted values log-variance (dims: num_steps x num_cells) (default= None)
        
        '''
        
        num_cells = pred.shape[-1]
        
        nrows = int(num_traces/ncols)
        fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
        axs = np.ravel(axs)
        
        if mode == 'rand':  
            idxs  = np.random.choice(list(range(num_cells)), size=num_traces, replace=False)
            idxs.sort()
                
        elif mode == 'activity':
            idxs = true.max(axis=0).argsort()[-num_traces:]
        
        else:
            idxs  = list(range(num_cells))
        
        time = np.arange(0, self.T*self.dt, self.dt)
        
        def zscore(x):
            return (x - x.mean())/x.std()
        
        if norm:
            ztrue = zscore(true)
            zpred = zscore(pred)
        else:
            ztrue = true
            zpred = pred
            
        zmin = min(zpred[:, idxs].min(), ztrue[:, idxs].min())
        zmax = max(zpred[:, idxs].max(), ztrue[:, idxs].max())
        
        if np.any(pred_logvar):
            pred_stdev = np.exp(0.5*pred_logvar)
            zmin = min(zmin, (zpred[:, idxs] - pred_stdev[:, idxs]).min())
            zmax = max(zmax, (zpred[:, idxs] + pred_stdev[:, idxs]).max())
        
        for ii, (ax,idx) in enumerate(zip(axs,idxs)):
            plt.sca(ax)
            plt.plot(time, zpred[:, idx], lw=2, color='#37A1D0')
            if np.any(pred_logvar):
                plt.fill_between(time, zpred[:, idx] + pred_stdev[:, idx], zpred[:, idx] - pred_stdev[:, idx], color= '#37A1D0', alpha=0.5, zorder=-2)
            plt.plot(time, ztrue[:, idx], lw=2, color='#E84924')
            plt.ylim(zmin-(zmax-zmin)*0.1, zmax+(zmax-zmin)*0.1)
            
            # Hide the right and top spines
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            if ii >= num_traces - ncols:
                plt.xlabel('time (s)', fontsize=14)
                plt.xticks(fontsize=12)
                ax.xaxis.set_ticks_position('bottom')
                
            else:
                plt.xticks([])
                ax.xaxis.set_ticks_position('none')
                ax.spines['bottom'].set_visible(False)

            if ii%ncols==0:
                plt.yticks(fontsize=12)
                ax.yaxis.set_ticks_position('left')
            else:
                plt.yticks([])
                ax.yaxis.set_ticks_position('none')
                ax.spines['left'].set_visible(False)
                
        fig.subplots_adjust(wspace=0.1, hspace=0.1)        
        return fig
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------ 

    
    def plot_inputs(self, fig_width=8, fig_height=1.5):
    
        figsize = (fig_width, fig_height*self.u_dim)
        fig, axs = plt.subplots(nrows=self.u_dim, figsize=figsize)
        fig.suptitle('Input to the generator for a sampled trial', y=1.2)
        inputs = self.inputs_mean.mean(dim=0).cpu().numpy()
        time = np.arange(0, self.T*self.dt, self.dt)
        for jx in range(self.u_dim):
            if self.u_dim > 1:
                plt.sca(axs[jx])
            else:
                plt.sca(axs)
            plt.plot(time, inputs[:, jx])
            plt.xlabel('time (s)')
        return fig

    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    def plot_rsquared(self, pred, true, figsize=(6, 4)):
        fig = plt.figure(figsize=figsize)
        var = true.var()
        mse = ((pred - true)**2).mean()
        rsq = 1-mse/var
        
        plt.plot(np.ravel(true), np.ravel(pred), '.')
        plt.xlabel('Ground Truth Rate (Hz)')
        plt.ylabel('Inferred Rates (Hz)')
        
        plt.title('R-squared coefficient = %.3f'%rsq)
        
        return fig
    
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    def plot_summary(self, data, truth=None, num_average=100):
        
        plt.close()
        
        figs_dict = {}
        
        batch_example, ix = batchify_random_sample(data, num_average)
        
        pred_data_mean, pred_data_logvar = self.reconstruct(batch_example)
        true_data = data[ix].cpu().numpy()
        figs_dict['traces'] = self.plot_traces(pred_data_mean, true_data, mode='activity', norm=False)
        figs_dict['traces'].suptitle('Fluorescence Data vs.Inferred Fluroscence Mean')
        figs_dict['traces'].legend(['Inferred Fluroscence Mean', 'Fluorescence Data'])
        
        if torch.is_tensor(truth):
            pred_rate = self.outputs_mean.mean(dim=0).cpu().numpy()
            true_rate = truth[ix].cpu().numpy()
            figs_dict['truth'] = self.plot_traces(pred_rate, true_rate, mode='rand')
            figs_dict['truth'].suptitle('Inferred vs. Ground-truth')
            figs_dict['truth'].legend(['Inferred', 'Ground-truth'])
            figs_dict['rsq']   = self.plot_rsquared(pred_rate, true_rate)
        
        figs_dict['inputs']  = self.plot_inputs()
        
        return figs_dict

    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    def save_checkpoint(self, force=False, purge_limit=50, output='.', name = 'best'):
        '''
        Save checkpoint of network parameters and izer state
        
        Arguments:
            force (bool) : force checkpoint to be saved (default = False)
            purge_limit (int) : delete previous checkpoint if there have been fewer
                                epochs than this limit before saving again
        '''
        
        # output_filename of format [timestamp]_epoch_[epoch]_loss_[training].pth:
        #  - timestamp   (YYMMDDhhmm)
        #  - epoch       (int)
        #  - loss        (float with decimal point replaced by -)
        
        save_loc = '%s/models/%s/%s/checkpoints/'%(output, self.dataset_name, self.run_name)

        if force:
            pass
        else:
            if purge_limit:
                # Get checkpoint filenames
                try:
                    _,_,filenames = list(os.walk(save_loc))[0]
                    split_filenames = [os.path.splitext(fn)[0].split('_') for fn in filenames]
                    epochs = [att[2] for att in split_filenames]
                    epochs.sort()
                    last_saved_epoch = epochs[-1]
                    if self.epochs - 50 <= int(last_saved_epoch):
                        rm_filename = [filename for filename in filenames if last_saved_epoch in filename][0]
                        os.remove(save_loc+rm_filename)
                    
                except IndexError:
                    pass

        # Get current time in YYMMDDhhmm format
        timestamp = datetime.datetime.now().strftime('%y%m%d%H%M')
        
        # Get epoch_num as string
        epoch = str('%i'%self.epochs)
        
        # Get training_error as string
        loss = str(self.valid_loss_store[-1]).replace('.','-')
        
        output_filename = name + '.pth'
        
        assert os.path.splitext(output_filename)[1] == '.pth', 'Output filename must have .pth extension'
                
        # Create dictionary of training variables
        train_dict = {'best' : self.best, 'train_loss_store': self.train_loss_store,
                      'valid_loss_store' : self.valid_loss_store,
                      'full_loss_store' : self.full_loss_store,
                      'epochs' : self.epochs, 'current_step' : self.current_step,
                      'last_decay_epoch' : self.last_decay_epoch,
                      'learning_rate' : self.learning_rate,
                      'cost_weights' : self.cost_weights,
                      'dataset_name' : self.dataset_name}
        
        # Save network parameters, optimizer state, and training variables
        torch.save({'net' : self.state_dict(), 'opt' : self.optimizer.state_dict(), 'train' : train_dict},
                   save_loc+output_filename)

    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    def load_checkpoint(self, input_filename='best', output='.'):
        '''
        Load checkpoint of network parameters and optimizer state

        required arguments:
            - input_filename (string): options:
                - path to input file. Must have .pth extension
                - 'best' (default) : checkpoint with lowest saved loss
                - 'recent'         : most recent checkpoint
                - 'longest'        : checkpoint after most training
            
            - dataset_name : if input_filename is not a path, must not be None
        '''
        # save_loc = '../%s/models/%s/%s/checkpoints/'%(output, self.dataset_name, self.run_name)
        save_loc = '../models/%s/%s/checkpoints/'%(self.dataset_name, self.run_name)
        
        print('save_loc:', save_loc)
        
        # If input_filename is not a filename, get checkpoint with specified quality (best, recent, longest)
        if not os.path.exists(input_filename):
            # Get checkpoint filenames
            try:
                _,_,filenames = list(os.walk(save_loc))[0]
            except IndexError:
                return
            
            if len(filenames) > 0:
            
                # Sort in ascending order
                filenames.sort()
                
                '''
                # Split filenames into attributes (dates, epochs, loss)
                split_filenames = [os.path.splitext(fn)[0].split('_') for fn in filenames]
                
                dates = [att[0] for att in split_filenames]
                epoch = [att[2] for att in split_filenames]
                loss  = [att[-1] for att in split_filenames]
                '''

                if input_filename == 'best':
                    # Get filename with lowest loss. If conflict, take most recent of subset.
                    input_filename = filenames[0]

                elif input_filename == 'recent':
                    # Get filename with most recent timestamp. If conflict, take first one
                    dates.sort()
                    recent = dates[-1]
                    input_filename = [fn for fn in filenames if recent in fn][0]

                elif input_filename == 'longest':
                    # Get filename with most number of epochs run. If conflict, take most recent of subset.
                    epoch.sort()
                    longest = epoch[-1]
                    input_filename = [fn for fn in filenames if longest in fn][-1]

                else:
                    assert False, 'input_filename must be a valid path, or one of \'best\', \'recent\', or \'longest\''
                    
            else:
                return

        assert os.path.splitext(input_filename)[1] == '.pth', 'Input filename must have .pth extension'

        # Load checkpoint
        print('Load checkpoint:', save_loc+input_filename)
        state = torch.load(save_loc+input_filename)

        # Set network parameters
        self.load_state_dict(state['net'])

        # Set optimizer state
        self.optimizer.load_state_dict(state['opt'])

        # Set training variables
        self.best                  = state['train']['best']
        self.train_loss_store      = state['train']['train_loss_store']
        self.valid_loss_store      = state['train']['valid_loss_store']
        self.full_loss_store       = state['train']['full_loss_store']
        self.epochs                = state['train']['epochs']
        self.current_step          = state['train']['current_step']
        self.last_decay_epoch      = state['train']['last_decay_epoch']
        self.learning_rate         = state['train']['learning_rate']
        self.cost_weights          = state['train']['cost_weights']
        self.dataset_name          = state['train']['dataset_name']
        
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    def health_check(self, writer):
        '''
        Checks the gradient norms for each parameter, what the maximum weight is in each weight matrix,
        and whether any weights have reached nan
        
        Report norm of each weight matrix
        Report norm of each layer activity
        Report norm of each Jacobian
        
        To report by batch. Look at data that is inducing the blow-ups.
        
        Create a -Nan report. What went wrong? Create file that shows data that preceded blow up, 
        and norm changes over epochs
        
        Theory 1: sparse activity in real data too difficult to encode
            - maybe, but not fixed by augmentation
            
        Theory 2: Edgeworth approximation ruining everything
            - probably: when switching to order=2 loss does not blow up, but validation error is huge
        '''
        
        hc_results = {'Weights' : {}, 'Gradients' : {}, 'Activity' : {}}
        odict = self._modules
        ii=1
        for name in odict.keys():
            if 'gru' in name:
                writer.add_scalar('3_Weight_norms/%ia_%s_ih'%(ii, name), odict.get(name).weight_ih.data.norm(), self.current_step)
                writer.add_scalar('3_Weight_norms/%ib_%s_hh'%(ii, name), odict.get(name).weight_hh.data.norm(), self.current_step)
                
                if self.current_step > 1:

                    writer.add_scalar('4_Gradient_norms/%ia_%s_ih'%(ii, name), odict.get(name).weight_ih.grad.data.norm(), self.current_step)
                    writer.add_scalar('4_Gradient_norms/%ib_%s_hh'%(ii, name), odict.get(name).weight_hh.grad.data.norm(), self.current_step)
            
            elif 'fc' in name or 'conv' in name:
                writer.add_scalar('3_Weight_norms/%i_%s'%(ii, name), odict.get(name).weight.data.norm(), self.current_step)
                if self.current_step > 1:
                    writer.add_scalar('4_Gradient_norms/%i_%s'%(ii, name), odict.get(name).weight.grad.data.norm(), self.current_step)
 
            ii+=1
        
        writer.add_scalar('5_Activity_norms/1_efgen', self.efcon.data.norm(), self.current_step)
        writer.add_scalar('5_Activity_norms/2_ebgen', self.ebcon.data.norm(), self.current_step)
        return
        
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    def _set_params(self, params):
        for k in params.keys():
            self.__setattr__(k, params[k])

    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    
    def _update_params(self, prev_params, new_params):
        if new_params:
            params = update_param_dict(prev_params, new_params)
        else:
            params = prev_params
        self._set_params(params)