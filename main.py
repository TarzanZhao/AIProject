import argparse
import Board
import torch
import Interface
import torch.nn
import Training
import Experiment
from Timer import timer
from DataStorage import dataProcessor
import Agent

def main(train):
    # ALl Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', type=int, default=2)
    parser.add_argument('--size', type=int, default=8)
    parser.add_argument('--numOfIterations', type=int, default=150)
    parser.add_argument('--numberForWin', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--drop_rate',type=float,default=0.3)
    parser.add_argument('--trainround', type=int, default=2)
    parser.add_argument('--trainepochs', type=int, default=500)
    parser.add_argument('--numOfEvaluations',type=int,default=5)
    parser.add_argument('--overwrite',type=int,default=1) # overwrite previous network
    parser.add_argument('--agentFirst',type=int,default=1) # agent or human play first
    args = parser.parse_args()
    args.device = ('cuda' if torch.cuda.is_available() else 'cpu')
    dataProcessor.initSimulator(Board.Board(args.size, args.numberForWin))

    if train==1:
        Training.Training(args)
    elif train==0:
        args.device = 'cpu'
        timer.clear()
        #Interface.Play(args,Interface.IntelxligenceAgent(args))
        Interface.Play(args, Agent.SearchAgent(4))
    else:
        exp = Experiment.Experiment(args)
        exp.selfplayWithDifferentNumOfIterations()


if __name__ == '__main__':
    # 2 experiment; 1 to train; 0 to visualize the game
    main(0)
