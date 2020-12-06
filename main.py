import argparse
import Board
import torch
import Interface
import torch.nn
import Training
import Experiment
import Game
from Timer import timer
from DataStorage import dataProcessor
import Agent
import random
import numpy as np
from Training import NetworkTraining

def main(train):
    # ALl Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--size', type=int, default=8)
    parser.add_argument('--numOfIterations', type=int, default=120)
    parser.add_argument('--numberForWin', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--drop_rate',type=float,default=0.3)
    parser.add_argument('--trainround', type=int, default=2)
    parser.add_argument('--trainepochs', type=int, default=500)
    parser.add_argument('--numOfEvaluations',type=int,default=5)
    parser.add_argument('--overwrite',type=int,default=1) # overwrite previous network
    parser.add_argument('--agentFirst',type=int,default=0) # agent or human play first
    parser.add_argument('--batchsize', type=int,default=256)
    args = parser.parse_args()
    args.device = ('cuda' if torch.cuda.is_available() else 'cpu')
    dataProcessor.initSimulator(Board.Board(args.size, args.numberForWin))

    if train==1:
        Training.Training(args)
    elif train==0:
        args.device = 'cpu'
        timer.clear()
        Interface.Play(args,Interface.IntelligenceAgent(args, 1))
#        Interface.Play(args, Agent.SearchAgent(3))
    elif train == 2:
        exp = Experiment.Experiment(args)
        exp.selfplayWithDifferentNumOfIterations(1)
    elif train == 3:
        lastk = 4
        numConfig = lastk + 47
        for k in range(lastk, numConfig):
            epsilon0 = round(random.uniform(0.2, 0.6), 2)
            epsilon1 = round(random.uniform(0.2, 0.6), 2)
            depth0 = np.random.choice(4, p=[0.05, 0.15, 0.2, 0.6])
            depth1 = np.random.choice(4, p=[0.05, 0.15, 0.2, 0.6])

            savePath = f"./searchPlayData/searchPlay-{k}"
            with open("./searchPlayData/configForEach", "a") as file:
                file.write(f"k={k} depth0={depth0} depth1={depth1} epsilon0={epsilon0} epsilon1={epsilon1}\n")
            print(f"\n\nk={k} depth0={depth0} depth1={depth1} epsilon0={epsilon0} epsilon1={epsilon1}\n----------------------START----------------------\n")

            with open("./logs/log.txt", "a") as file:
                file.write(f"k={k} depth0={depth0} depth1={depth1} epsilon0={epsilon0} epsilon1={epsilon1}\n")

            searchAgent0 = Agent.SearchAgent(depth0, epsilon=epsilon0)
            searchAgent1 = Agent.SearchAgent(depth1, epsilon=epsilon1)
            board = Board.Board(args.size, args.numberForWin)
            game = Game.Game(searchAgent0, searchAgent1, board)

            agent0Win = 0
            finalDataList = []
            totalGames = 200
            for ite in range(totalGames):
                try:
                    t = game.run() == searchAgent0
                    data0 = searchAgent0.dataList
                    data1 = searchAgent1.dataList
                    z = 1 if t else -1
                    j = 0
                    for i in range(len(data0)):
                        d0 = data0[i]
                        finalDataList.append((d0[0], d0[1], z))
                        if j < len(data1):
                            d1 = data1[j]
                            finalDataList.append((d1[0], d1[1], -z))
                            j += 1
                    finalDataList.append("end")
                    print(f"iteration {ite + 1}: search agent-{int(not t)} win the game")
                    if (ite+1) % 50 == 0:
                        dataProcessor.saveData(finalDataList, savePath)
                    agent0Win += t
                except:
                    with open("./logs/log.txt", "a") as file:
                        file.write(f"error occurs in iteration {k}\n")
            dataProcessor.saveData(finalDataList, savePath)
            print(f"search agent0 win {agent0Win} games out of {totalGames}")
            with open("./logs/log.txt", "a") as file:
                file.write(f"search agent0 win {agent0Win} games out of {totalGames}\n")
    elif train==4:
        totalDataList = []
        for k in range(51):
            file = f"./searchPlayData/searchPlay-{k}"
            dataList = dataProcessor.retrieveData(file)
            totalDataList.extend(dataList)
        currentModel = 0
        trainWorker = NetworkTraining(args)
        trainWorker.train(args.trainepochs, currentModel, totalDataList)




if __name__ == '__main__':
    # 2 experiment; 1 to train; 0 to visualize the game; 3: sample data from search agent and gready agent. 4:train model using selfplay data.
    main(0)
