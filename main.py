import argparse
import Agent
import PolicyValueFn
import Board
import Game
import torch
import Interface
import torch.nn
import os
import DataStorage
import Training


def main(train):
    # Training setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--size', type=int, default=8)
    parser.add_argument('--numOfIterations', type=int, default=100)
    parser.add_argument('--numberForWin', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--trainround', type=int, default=30)
    parser.add_argument('--trainepochs', type=int, default=50)
    parser.add_argument('--overwrite',type=int,default=0)
    args = parser.parse_args()
    args.device = ('cuda:1' if torch.cuda.is_available() else 'cpu')

    if train:
        Training.Training(args)
    else:
        args.device = 'cpu'
        Interface.Play(args)

if __name__ == '__main__':
    main(1)
