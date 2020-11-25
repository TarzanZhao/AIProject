import argparse
import Board
import torch
import Interface
import torch.nn
import Training
from DataStorage import dataProcessor


def main(train):
    # ALl Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--size', type=int, default=6)
    parser.add_argument('--numOfIterations', type=int, default=200)
    parser.add_argument('--numberForWin', type=int, default=3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--drop_rate',type=float,default=0.3)
    parser.add_argument('--trainround', type=int, default=1)
    parser.add_argument('--trainepochs', type=int, default=500)
    parser.add_argument('--overwrite',type=int,default=0) # overwrite previous network
    parser.add_argument('--agentFirst',type=int,default=1) # agent or human play first
    args = parser.parse_args()
    args.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataProcessor.initSimulator(Board.Board(args.size, args.numberForWin))

    if train:
        Training.Training(args)
    else:
        args.device = 'cpu'
        Interface.Play(args)

if __name__ == '__main__':
    # 1 to train; 0 to visualize the game
    main(1)
