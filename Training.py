from DataStorage import retrieveData,getLatestNetworkID
import torch
import Agent
import Board
import Game
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PolicyValueFn import PolicyValueFn
from PolicyValueFn import MixLoss
import os
import copy

class MyDataset(Dataset):
    def __init__(self, dataList):
        self.len = len(dataList)
        self.dataList = dataList
        # data augmentation
        self.map = []
        self.size = self.dataList[0][0].size()[1]
        size = self.size
        to = [i for i in range(0,size**2)]

        for flip in range(2):
            for rotate in range(4):

                m = torch.tensor(to).reshape((size, size))
                if flip==1:
                    m = m.flip(-1)
                m = torch.rot90(m, k=rotate)

                mp = [0 for i in range(size**2)]
                for x in range(size):
                    for y in range(size):

                        v = m[x][y]
                        mp[v.item()] = x*size+y
                self.map.append(mp)



    def __getitem__(self, index):
        id = index//8
        trans = index%8
        
        Y = torch.zeros(self.size**2)
        for i, v in enumerate(self.dataList[id][1]):
            Y[self.map[trans][i]] = v
        
        X = self.dataList[id][0]
        if trans>=4:
            X = X.flip(-1)
            trans-=4
        X = torch.rot90(X, k=trans, dims = (1,2))
        return X, (Y, self.dataList[id][2])

    def __len__(self):
        return self.len*8

class NetworkTraining:
    def __init__(self,arg):
        self.arg = arg
        pass

    def train(self, numOfEpoch, currentModel):
        file = f"./selfplay/selfplay-{currentModel}.txt"
        dataList = retrieveData(file)
        trainSet = MyDataset(dataList)
        dataloader = DataLoader(dataset=trainSet,
                                batch_size=256,
                                num_workers=4,
                                pin_memory=True,
                                shuffle=True
                                )
        criterion = MixLoss()

        network = PolicyValueFn(self.arg)
        if currentModel > 0:
            network.load_state_dict(torch.load(f'network/network-{currentModel - 1}.pt'))

        network.to(self.arg.device)
        optimizer = optim.SGD(
            network.parameters(), self.arg.lr
        )
        trainLossFile = "trainingloss/trainingloss-" + str(currentModel)
        if os.path.exists(trainLossFile):
            os.remove(trainLossFile)
            print("successfully deleted " + trainLossFile)

        for epoch in range(1, numOfEpoch+1):
            total_loss = 0
            loss = 0
            cnt = 0
            for i, batch in enumerate(dataloader):
                x, y1, y2 = batch[0].to(self.arg.device), batch[1][0].to(self.arg.device), batch[1][1].to(self.arg.device)
                z1, z2 = network(x)
                loss = criterion(z1, y1, z2, y2)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cnt += 1
                if cnt % 100 == 0:
                    with open(trainLossFile, mode="a") as file:
                        file.write("i={} loss={:.6f}".format(i, loss.item()))
            total_loss = total_loss / cnt
            print("epoch={} average_loss={:.6f}".format(epoch, total_loss))
            with open(trainLossFile, mode="a") as file:
                file.write("epoch={} average_loss={:.6f}".format(epoch, loss))

            if epoch % 10 == 0:
                torch.save(network.state_dict(), f"network/network-{currentModel}.pt")
        torch.save(network.state_dict(),f"network/network-{currentModel}.pt")

def Training(args):
    currentModel = -1 if args.overwrite else getLatestNetworkID()
    trainWorker = NetworkTraining(args)

    for rd in range(1, args.trainround + 1):
        print("round:%d" % rd)
        model = PolicyValueFn(args).to(args.device)
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

