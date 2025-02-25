# Low Rank Autoregressive Model Regularization and Tuning

Follow up experimentation from:
[<b>Active Learning of Neural Population Dynamics using two-photon holographic optogenetics</b>](https://arxiv.org/abs/2412.02529)

There is a strong need for techniques that minimize the amount of data needed to learn neural population dynamics due to experimental time and resource constraints. "Active learning of neural population dynamics using two-photon holographic optogenetics" is a NeurIPS 2024 paper that attempts to determine the most effective photostimulation patterns for identifying neural population dynamics. The goal is to select the neurons that have the most informative neural responses to inform a dynamical model of the neural population activity.

The project’s active learning technique takes advantage of the low-rank structure of the neural population dynamics to determine the most informative photostimulation patterns. It uses SVD to create low rank autoregressive models that predict neural activity. I conducted regularization and hyperparameter tuning experiments on these models. This resulted in 15-18% improvements (MSE) in model performance.

![](media/lowrank_reg.png)


# Environment + Setup

To begin on your own machine, clone this repository locally
```bash
git clone https://github.com/andrewjjeon/lowrank.git
```
Install requirements:
```bash
$ conda create -n lowrank python=3.8 -y
$ conda activate lowrank
$ pip install -r requirements.txt
$ cd 3dvlmaps
```

After environment setup,
- Get public BCI_xx_xxxxxx.npy data from Active Learning paper authors.
- Run the python scripts in aj_models in this order --> bci_data_processing.py --> furtherprocessing.py --> closedform.py --> train_model.py --> load_model.py

- Use config to control parameters