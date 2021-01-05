import argparse
import torch
import os

from typing import Tuple, List

args = None

def initialize_args():
    # ALl Hyper Parameters
    parser = argparse.ArgumentParser()

    parser.add_argument('--todo',choices=['visualize', 'selfplaytrain', 'experiment', 'sampledata', 'supervisedtrain'], default='visualize')
    parser.add_argument('--visualize',choices=['network', 'intelligent', 'minmax'], default='network')

    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--numOfIterations', type=int, default=150)
    parser.add_argument('--numberForWin', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--drop_rate', type=float, default=0.3)
    parser.add_argument('--trainround', type=int, default=50)
    parser.add_argument('--trainepochs', type=int, default=50)
    parser.add_argument('--numOfEvaluations', type=int, default=1)
    parser.add_argument('--overwrite', type=int, default=0)  # overwrite previous network
    parser.add_argument('--agentFirst', type=int, default=1)  # agent or human play first
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--miniTrainingEpochs', type=int, default=10)
    parser.add_argument('--buffersize', type=int, default=256)
    parser.add_argument('--openReplayBuffer', type=bool, default=1)
    parser.add_argument('--maxBufferSize', type=int, default=4096)
    parser.add_argument('--rolloutMode',choices=['network', 'random', 'minmax', 'mix_random', 'mix_minmax', 'greedy'], default='network')
    parser.add_argument('--balance', type=float, default=0.8)

    parser.add_argument('--sampleRound', type=int, default=20)
    parser.add_argument('--sampleSize', type=int, default=500)
    parser.add_argument('--epsilon0', type=Tuple[float], default=(0.1, 0.2))
    parser.add_argument('--epsilon1', type=Tuple[float], default=(0.1, 0.2))
    parser.add_argument('--probDepth0', type=Tuple[float], default=(0.0,0.0,0.5,0.5) )
    parser.add_argument('--probDepth1', type=Tuple[float], default=(0.2,0.3,0.2,0.3) )

    parser.add_argument('--n_log_step', type=int, default=15)
    parser.add_argument('--n_save_step', type=int, default=50)

    parser.add_argument('--n_train_data', type=int, default=3)
    parser.add_argument('--modelID', type=int, default=0)

    parser.add_argument('--log_root', type=str,default='./logs')
    parser.add_argument('--figure_root', type=str,default='./figure')
    parser.add_argument('--model_root', type=str,default='./checkpoints')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--save_folder', type=str)
    parser.add_argument('--load_data_folder', type=str, default='nofolder')

    global args
    args = parser.parse_args()
    args.device = ('cuda' if torch.cuda.is_available() else 'cpu')

    setattr(args, 'log_folder', os.path.join(args.log_root, args.save_folder))
    setattr(args, 'figure_folder', os.path.join(args.figure_root, args.save_folder))
    setattr(args, 'model_folder', os.path.join(args.model_root, args.save_folder))
    if args.todo == 'supervisedtrain':
        assert args.load_data_folder != 'nofolder', 'not specify from which folder to load data.'
        setattr(args, 'data_folder', os.path.join(args.data_root, args.load_data_folder))
    else:
        setattr(args, 'data_folder', os.path.join(args.data_root, args.save_folder))

    makedirs(args.log_folder)
    if not args.todo in ['visualize', 'sampledata']:
        makedirs(args.model_folder)
    if not args.todo in ['visualize', 'supervisedtrain']:
        makedirs(args.data_folder)
    makedirs(args.figure_folder)

def get_args():
    return args

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        return False