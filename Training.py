from DataStorage import dataProcessor
import torch
import Agent
import Board
import Game
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PolicyValueFn import PolicyValueFn
from PolicyValueFn import MixLoss
from Timer import timer
import os
import math
from Experiment import Experiment
import random


class MyDataset(Dataset):
    def __init__(self, dataList):
        self.len = len(dataList)
        self.dataList = dataList
        # data augmentation
        self.map = []
        self.size = self.dataList[0][0].size()[1]
        size = self.size
        to = [i for i in range(0, size ** 2)]

        for flip in range(2):
            for rotate in range(4):

                m = torch.tensor(to).reshape((size, size))
                if flip == 1:
                    m = m.flip(-1)
                m = torch.rot90(m, k=rotate)

                mp = [0 for i in range(size ** 2)]
                for x in range(size):
                    for y in range(size):
                        v = m[x][y]
                        mp[v.item()] = x * size + y
                self.map.append(mp)

    def __getitem__(self, index):
        id = index // 8
        trans = index % 8

        Y = torch.zeros(self.size ** 2)
        for i, v in enumerate(self.dataList[id][1]):
            Y[self.map[trans][i]] = v

        X = self.dataList[id][0]
        if trans >= 4:
            X = X.flip(-1)
            trans -= 4
        X = torch.rot90(X, k=trans, dims=(1, 2))
        return X, (Y, self.dataList[id][2])

    def __len__(self):
        return self.len * 8


class NetworkTraining:
    def __init__(self, arg):
        self.arg = arg
        pass

    def getReplayData(self, currentModel, dataList):
        dataWithLoss = []
        with torch.no_grad():
            dataList = MyDataset(dataList)
            dataloader = DataLoader(dataset=dataList,
                                    batch_size=1,
                                    num_workers=4,
                                    pin_memory=True,
                                    shuffle=False
                                    )
            criterion = MixLoss()
            network = PolicyValueFn(self.arg)
            if currentModel > 0:
                network.load_state_dict(torch.load(f'network/network-{currentModel}.pt'))

            network.to(self.arg.device)

            for batch in dataloader:
                x, y1, y2 = batch[0].to(self.arg.device), batch[1][0].to(self.arg.device), batch[1][1].to(
                    self.arg.device)
                z1, z2 = network(x)
                loss = criterion(z1, y1, z2, y2)
                dataWithLoss.append((loss, (x[0].to('cpu'), y1[0].to('cpu'), y2[0].to('cpu'))))
            dataWithLoss.sort(key=lambda pair: pair[0], reverse=True)
            replayData = []
            for i in range(min(len(dataWithLoss), self.arg.buffersize)):
                replayData.append(dataWithLoss[i][1])
        return replayData

    def train(self, numOfEpoch, currentModel, dataList=None, update=True):
        if dataList is None:
            file = f"./selfplay/selfplay-{currentModel}.txt"
            dataList = dataProcessor.retrieveData(file)
        trainSet = MyDataset(dataList)
        dataloader = DataLoader(dataset=trainSet,
                                batch_size=self.arg.batchsize,
                                num_workers=4,
                                pin_memory=True,
                                shuffle=True
                                )
        criterion = MixLoss()

        network = PolicyValueFn(self.arg)
        x = 1 if update else 0
        if currentModel - x >= 0:
            print(f"load network: {currentModel-x}")
            network.load_state_dict(torch.load(f'network/network-{currentModel - x}.pt',map_location=torch.device(self.arg.device)))

        network.to(self.arg.device)
        optimizer = optim.Adam(network.parameters())
        trainLossFile = "trainingloss/trainingloss-" + str(currentModel)
        if os.path.exists(trainLossFile):
            os.remove(trainLossFile)
            print("successfully deleted " + trainLossFile)

        network.train()
        total_loss = 0
        for epoch in range(1, numOfEpoch + 1):
            loss = 0
            cnt = 0
            for i, batch in enumerate(dataloader):
                x, y1, y2 = batch[0].to(self.arg.device), batch[1][0].to(self.arg.device), batch[1][1].to(
                    self.arg.device)
                z1, z2 = network(x)
                loss = criterion(z1, y1, z2, y2)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cnt += 1
                if cnt % 100 == 0:
                    with open(trainLossFile, mode="a") as file:
                        file.write("i={} loss={:.6f}\n".format(i, loss.item()))
            total_loss = total_loss / cnt
            print("epoch={} average_loss={:.6f}".format(epoch, total_loss))
            with open(trainLossFile, mode="a") as file:
                file.write("epoch={} average_loss={:.6f}\n".format(epoch, loss))

            if epoch % 10 == 0:
                torch.save(network.state_dict(), f"network/network-{currentModel}.pt")
        torch.save(network.state_dict(), f"network/network-{currentModel}.pt")
        print(f"save network: {currentModel}")
        return total_loss


def Training(args):
    currentModel = -1 if args.overwrite else dataProcessor.getLatestNetworkID()
    trainWorker = NetworkTraining(args)
    replayBuffer = []
    Loss = []
    WinRate = []

    for rd in range(1, args.trainround + 1):
        print("round:%d" % rd)
        if currentModel != -1:
            model = dataProcessor.loadNetwork(args, currentModel)
        else:
            model = PolicyValueFn(args).to(device=args.device)
        eta = math.log(args.trainround / rd) + 1
        file = f"./selfplay/selfplay-{currentModel + 1}.txt"
        agent1 = Agent.SelfplayAgent(args.numOfIterations, model, file, eta)
        b = Board.Board(args.size, args.numberForWin)
        g = Game.Game(agent0=agent1, agent1=agent1, simulator=b)

        for i in range(1, args.epochs + 1):
            print("epoch %d" % i)
            TimeID = timer.startTime("play time")
            g.run()
            timer.endTime(TimeID)
            timer.showTime(TimeID)
            if i % 25 == 0:
                agent1.saveData()
            if args.openReplayBuffer and len(replayBuffer)>args.buffersize:
                buffer = []
                for i in range(args.buffersize):
                    buffer.append(random.choice(replayBuffer))
                trainWorker.train(args.miniTrainingEpochs,currentModel,buffer,update=False)
            #if args.openReplayBuffer and len(replayBuffer):
            #    trainWorker.train(args.miniTrainingEpochs, currentModel, replayBuffer, update=False)
        agent1.saveData()
        dataList = dataProcessor.retrieveData(file)
        replayBuffer = replayBuffer + dataList
        if len(replayBuffer)>args.maxBufferSize:
            replayBuffer = replayBuffer[-args.maxBufferSize:]
        currentModel += 1
        TimeID = timer.startTime("network training")
        Loss.append(trainWorker.train(args.trainepochs, currentModel, dataList))
        timer.endTime(TimeID)
        timer.showTime(TimeID)

        #if args.openReplayBuffer:
        #    TimeID = timer.startTime("update replay buffer")
        #    replayBuffer = trainWorker.getReplayData(currentModel, dataList)
        #    timer.endTime(TimeID)
        #    timer.showTime(TimeID)
        agentTest = Agent.IntelligentAgent(args.numOfIterations, dataProcessor.loadNetwork(args))

        exp = Experiment(args)
        WinRate.append(exp.evaluationWithBaseLine(agentTest))
        print("WinRate: %.3f" %WinRate[-1])
    return Loss, WinRate


# compare current and previous network
def Evaluation(args):
    currentModel = dataProcessor.getLatestNetworkID()
    model1 = dataProcessor.loadNetwork(args, currentModel)
    model2 = dataProcessor.loadNetwork(args, currentModel - 1)
    agent1 = Agent.IntelligentAgent(2 * args.numOfIterations, model1)
    agent2 = Agent.IntelligentAgent(2 * args.numOfIterations, model2)
    board = Board.Board(args.size, args.numberForWin)
    game = Game.Game(agent1, agent2, board)
    Score = {agent1: 0, agent2: 1}
    print("First Player Case:")
    for i in range(1, 1 + args.numOfEvaluations):
        winner = game.run()
        Score[winner] += 1
        if winner == agent1:
            print("The %dth game: Win!" % i)
        else:
            print("The %dth game: Lose!" % i)
    game = Game.Game(agent2, agent1, board)
    print("Second Player Case:")
    for i in range(1, 1 + args.numOfEvaluations):
        winner = game.run()
        Score[winner] += 1
        if winner == agent1:
            print("The %dth game: Win!" % (i + args.numOfEvaluations))
        else:
            print("The %dth game: Lose!" % (i + args.numOfEvaluations))
    return Score[agent1] / Score[agent2]
