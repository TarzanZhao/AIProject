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
import argument
import logger
import os
from argument import makedirs
import traceback


def train(args, logger, dataProcessor):
    print("-2")
    Loss, WinRate = Training.Training()
    print("-1")
    exp = Experiment.Experiment()
    exp.simplePlot(range(args.trainround), Loss, "Training Loss Curve", xlabel='Training Rounds',
                   ylabel='Training Loss', color='blue', linestyle='-.')
    exp.simplePlot(range(args.trainround), WinRate, "Winning Rate Curve", xlabel='Training Rounds',
                   ylabel='Winning Rate', color='blue', linestyle='-.')
    logger.info("Loss and Winning Rate")
    loss_log = ''
    for i in Loss:
        loss_log += str(i) + ','
    logger.info(loss_log)
    loss_log = ''
    for i in WinRate:
        loss_log += str(i) + ','
    logger.info(loss_log)

def visualize(args, logger, dataProcessor):
    # model.state_dict().
    args.device = 'cpu'
    timer.clear()
    # Interface.Play(args,Interface.NetworkAgent(args))
    Interface.Play(args, Interface.IntelligenceAgent(args, args.modelID))
    #Interface.Play(args, Agent.SearchAgent(3,epsilon=0))


def experiment(args, logger, dataProcessor):
    exp = Experiment.Experiment()
    # agent = Interface.IntelligenceAgent(args)
    # agent2 = Agent.SearchAgent(4)
    # logger.info("The win rate for Network: %.3f" %exp.evaluation(agent,agent2))
    X, Y = exp.playWithBaselineInDifferentNumOfIterations()
    exp.simplePlot(X, Y, title="Wining Strategy with Different Tree Iteration")

def sampledata(args, logger, dataProcessor):
    for k in range(args.sampleRound):
        savePath = os.path.join(args.data_folder, 'searchPlay-' + str(k))

        epsilon0 = round(random.uniform(args.epsilon0[0], args.epsilon0[1]), 2)
        epsilon1 = round(random.uniform(args.epsilon1[0], args.epsilon1[1]), 2)
        depth0 = np.random.choice(4, p=list(args.probDepth0))
        depth1 = np.random.choice(4, p=list(args.probDepth1))

        logger.info(f"k={k} depth0={depth0} depth1={depth1} epsilon0={epsilon0} epsilon1={epsilon1} "
                    f"\n------------------------START------------------------")

        searchAgent0 = Agent.SearchAgent(depth0, epsilon=epsilon0)
        searchAgent1 = Agent.SearchAgent(depth1, epsilon=epsilon1)
        board = Board.Board(args.size, args.numberForWin)
        game = Game.Game(searchAgent0, searchAgent1, board)

        agent0Win = 0
        finalDataList = []
        totalGames = args.sampleSize
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
                if ite % args.n_log_step == 0:
                    logger.info(f"iteration {ite + 1}: search agent-{int(not t)} win the game")
                if (ite + 1) % args.n_save_step == 0:
                    dataProcessor.saveData(finalDataList, savePath)
                agent0Win += t
            except ArithmeticError:
                logger.info(f"error occurs in iteration {k}")
        dataProcessor.saveData(finalDataList, savePath)
        logger.info(f"search agent0 win {agent0Win} games out of {totalGames}")

def supervisedtrain(args, logger, dataProcessor):
    totalDataList = []
    for k in range(args.n_train_data):
        file = os.path.join(args.data_folder, 'searchPlay-' + str(k))
        dataList = dataProcessor.retrieveData(file)
        totalDataList.extend(dataList)
    currentModel = 0
    trainWorker = NetworkTraining()
    trainWorker.train(args.trainepochs, currentModel, totalDataList)


if __name__ == '__main__':
    argument.initialize_args()
    args = argument.get_args()
    logger.initialize_logger(args.log_folder, args.todo, 'info')
    logger = logger.get_logger()
    timer.init()

    dataProcessor.initSimulator(Board.Board(args.size, args.numberForWin))

    if args.todo == 'selfplaytrain':
        train(args, logger, dataProcessor)
    elif args.todo == 'visualize':
        visualize(args, logger, dataProcessor)
    elif args.todo == 'experiment':
        experiment(args, logger, dataProcessor)
    elif args.todo == 'sampledata':
        sampledata(args, logger, dataProcessor)
    elif args.todo == 'supervisedtrain':
        supervisedtrain(args, logger, dataProcessor)

