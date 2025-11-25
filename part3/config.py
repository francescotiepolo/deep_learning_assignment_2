import argparse

import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['gcn', 'matrix-gcn', 'gat'])
    return parser.parse_args()

def get_config():
    config = parse_args()
    config.num_epochs = 50

    config.n_layers = 2
    config.dropout = 0.1
    config.hidden_dim = 124

    config.lr = 1e-3
    config.weight_decay = 1e-5

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        config.device = torch.device("mps")
    elif torch.cuda.is_available():
        config.device = torch.device("cuda")
    else:
        config.device = torch.device("cpu")
    
    config.progress_bar = True
    config.log_dir = './logs'

    config.seed = 42
    return config