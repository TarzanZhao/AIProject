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
    parser.add_argument('--numOfIterations', type=int, default=40)
    parser.add_argument('--numberForWin', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--trainround', type=int, default=10)
    parser.add_argument('--trainepochs', type=int, default=50)
    parser.add_argument('--overwrite',type=int,default=1)
    #    parser.add_argument('--modelid', type=int)
    args = parser.parse_args()
    args.device = ('cuda:1' if torch.cuda.is_available() else 'cpu')

    # 寻找最新的神经网络
    path = "./network"
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    currentModel = -1
    if not args.overwrite:
        for file in files:
            if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
                filestr = file.split("/")[-1]
                print(filestr)
                if filestr.startswith("network-"):
                    idstr = filestr[8:-3]
                    print(idstr)
                    currentModel = max(currentModel, int(idstr))
    trainWorker = Training(args)

    # model = PolicyValueFn.PolicyValueFn(args).to(args.device)
    for rd in range(1, args.trainround+1):
        print("round:%d" % rd)
        model = PolicyValueFn.PolicyValueFn(args).to(args.device)
        if currentModel != -1:
            model.load_state_dict(torch.load(f'network/network-{currentModel}.pt'))
        agent1 = Agent.SelfplayAgent(args.numOfIterations, model, f"selfplay/selfplay-{currentModel + 1}.txt")
        b = Board.Board(args.size, args.numberForWin)
        g = Game.Game(agent0=agent1, agent1=agent1, simulator=b)

        for i in range(1, args.epochs + 1):
            print("epoch %d" % i)
            g.run()
            if i % 25 == 0:
                agent1.saveData()
        agent1.saveData()

        currentModel += 1
        trainWorker.train(args.trainepochs, currentModel)


if __name__ == '__main__':
    main()
