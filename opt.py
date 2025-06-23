# Configuration and Hyperparameter Management for FC-HGNN
# Handles command-line arguments, device setup, and reproducible random seeds

import datetime
import argparse
import random
import numpy as np
import torch

class OptInit():
    # Configuration initialization class for FC-HGNN model
    
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch implementation of FC-HGNN')
        
        # Training/evaluation mode
        parser.add_argument('--train', default=1, type=int, help='train(default) or evaluate')
        parser.add_argument('--use_cpu', action='store_true', help='use cpu?')
        
        # Model hyperparameters
        parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
        parser.add_argument('--wd', default=5e-5, type=float, help='weight decay')
        parser.add_argument('--num_iter', default= 100, type=int, help='number of epochs for training')
        parser.add_argument('--dropout', default=0.3, type=float, help='ratio of dropout')
        parser.add_argument('--batch_size', type=int, default=100, help='batch size for potential mini-batching (0 = full graph)')
        parser.add_argument('--amp', action='store_true', help='enable automatic mixed precision (AMP)')
        parser.add_argument('--num_classes', type=int, default=3, help='number of classes')
        parser.add_argument('--n_folds', type=int, default=5, help='number of folds')
        
        # File paths
        parser.add_argument('--ckpt_path', type=str, default='./checkpoints/', help='checkpoint path to save trained models')
        parser.add_argument('--log_path', type=str, default=r'./checkpoints/inffus_log.txt', help='the path of the log')
        parser.add_argument('--subject_IDs_path', type=str, default='./subjects.txt', help='the path of the subject_IDs')
        parser.add_argument('--phenotype_path', type=str, default=r"./data/TADPOLE_COMPLETE.csv", help='the path of the phenotype data')
        parser.add_argument('--data_path', type=str, default=r'./data/FC_Matrices',help='the path of the data')

        # Graph construction parameters
        parser.add_argument('--alpha', default=0.65, type=float, help='adjacency threshold set when building  Brain_connectomic_graph')
        parser.add_argument('--beta', default=1.5, type=float, help='adjacency threshold set when building HPG')
        parser.add_argument('--k1', default=0.9, type=float, help='the pooling ratio of the channel 1 of the LGP')
        parser.add_argument('--k2', default=0.5, type=float,help='the pooling ratio of the channel 2 of the LGP')

        args = parser.parse_args()

        # Add timestamp for experiment tracking
        args.time = datetime.datetime.now().strftime("%y%m%d")

        # Device configuration (GPU/CPU)
        if not args.use_cpu and torch.cuda.is_available():
            args.device = torch.device('cuda:0')
        else:
            args.device = torch.device('cpu')

        self.args = args

    def print_args(self):
        # Print all configuration parameters for logging/debugging
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("==========     CONFIG END    =============")
        print("\n")
        phase = 'train' if self.args.train==1 else 'eval'
        print('===> Phase is {}.'.format(phase))

    def initialize(self):
        # Initialize configuration with reproducible random seeds
        self.set_seed(666)
        self.print_args()
        return self.args

    def set_seed(self, seed=0):
        # Set random seeds for reproducible results across multiple runs
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enable fast algorithms (Are non-deterministic)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


