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

    path = "./models" #文件夹目录
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



    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    trainWorker = Training(args)

    for rd in range(args.trainround): 
        model = PolicyValueFn.PolicyValueFn(args).to(device)
        if currentModel != -1:
            model.load_state_dict(torch.load(f'./network-{currentModel}.pt'))
        agent1 = Agent.SelfplayAgent(args.numOfIterations, model, f"selfPlay-{currentModel+1}.txt")
        b = Board.Board(args.size, args.numberForWin)
        g = Game.Game(agent0=agent1, agent1=agent1, simulator=b)

        for i in range(1, args.epochs + 1):
            print("epoch %d" % i)
            g.run()
            if i % 20 == 0:
                agent1.saveData()
        agent1.saveData()

        currentModel += 1
        trainWorker.train(args.trainepochs, currentModel)

if __name__ == '__main__':
    main()
