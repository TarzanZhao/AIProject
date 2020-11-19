import argparse
import Agent
import PolicyValueFn
import Board
import Game
import torch
import torch.nn
import os
from Training import Training

def main():
    # Training setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-batch-size', type=int, default=15000)
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--size', type=int, default=8)
    parser.add_argument('--numOfIterations', type=int, default=5)
    parser.add_argument('--numberForWin', type=int, default=4)
    parser.add_argument('--device',type=str,default='cpu')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--show-size', type=int, default=15000)
    parser.add_argument('--std', type=float, default=0.01)
    parser.add_argument('--show', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--trainround', type=int, default=1)
    parser.add_argument('--trainepochs', type=int, default=50)
#    parser.add_argument('--modelid', type=int)
    args = parser.parse_args()

    path = "./"
    files= os.listdir(path) #得到文件夹下的所有文件名称
    currentModel = -1
    for file in files: 
        if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
            filestr = file.split("/")[-1]
            print(filestr)
            if filestr.startswith("network-"):
                idstr = filestr[9:-3]
                print(idstr)
                currentModel = max(currentModel, int(idstr))


    args.device = ('cuda' if torch.cuda.is_available() else 'cpu')
    trainWorker = Training(args)
    currentModel+=1
    trainWorker.train(args.trainepochs, currentModel)



if __name__ == '__main__':
    main()
