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

    def train(self, numOfEpoch, currentModel, dataList = None):
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
        if currentModel > 0:
            network.load_state_dict(torch.load(f'network/network-{currentModel - 1}.pt'))

        network.to(self.arg.device)
        optimizer = optim.Adam(network.parameters())
        trainLossFile = "trainingloss/trainingloss-" + str(currentModel)
        if os.path.exists(trainLossFile):
            os.remove(trainLossFile)
            print("successfully deleted " + trainLossFile)

        network.train()
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
                        file.write("i={} loss={:.6f}\n".format(i, loss.item()))
            total_loss = total_loss / cnt
            print("epoch={} average_loss={:.6f}".format(epoch, total_loss))
            with open(trainLossFile, mode="a") as file:
                file.write("epoch={} average_loss={:.6f}\n".format(epoch, loss))

            if epoch % 10 == 0:
                torch.save(network.state_dict(), f"network/network-{currentModel} .pt")
        torch.save(network.state_dict(),f"network/network-{currentModel}.pt")

def Training(args):
    currentModel = -1 if args.overwrite else dataProcessor.getLatestNetworkID()
    trainWorker = NetworkTraining(args)

    for rd in range(1, args.trainround + 1):
        print("round:%d" % rd)
        if currentModel != -1:
            model = dataProcessor.loadNetwork(args,currentModel)
        else:
            model = PolicyValueFn(args).to(device=args.device)
        agent1 = Agent.SelfplayAgent(args.numOfIterations, model, f"selfplay/selfplay-{currentModel + 1}.txt")
        b = Board.Board(args.size, args.numberForWin)
        g = Game.Game(agent0=agent1, agent1=agent1, simulator=b)

        for i in range(1, args.epochs + 1):
            print("epoch %d" % i)
            TimeID = timer.startTime("a single play")
            g.run()
            timer.endTime(TimeID)
            timer.showTime(TimeID)
            if i % 25 == 0:
                agent1.saveData()
        agent1.saveData()
        currentModel += 1
        TimeID = timer.startTime("network training")
        trainWorker.train(args.trainepochs, currentModel)
        timer.endTime(TimeID)
        timer.showTime(TimeID)

# compare current and previous network
def Evaluation(args):
    currentModel = dataProcessor.getLatestNetworkID()
    model1 = dataProcessor.loadNetwork(args, currentModel)
    model2 = dataProcessor.loadNetwork(args, currentModel-1)
    agent1 = Agent.IntelligentAgent(2*args.numOfIterations, model1)
    agent2 = Agent.IntelligentAgent(2*args.numOfIterations,model2)
    board = Board.Board(args.size,args.numberForWin)
    game = Game.Game(agent1,agent2,board)
    Score = {agent1:0,agent2:1}
    print("First Player Case:")
    for i in range(1,1+args.numOfEvaluations):
        winner = game.run()
        Score[winner] +=1
        if winner==agent1:
            print("The %dth game: Win!"%i)
        else:
            print("The %dth game: Lose!"%i)
    game = Game.Game(agent2,agent1,board)
    print("Second Player Case:")
    for i in range(1,1+args.numOfEvaluations):
        winner = game.run()
        Score[winner] +=1
        if winner==agent1:
            print("The %dth game: Win!"%(i+args.numOfEvaluations))
        else:
            print("The %dth game: Lose!"%(i+args.numOfEvaluations))
    return Score[agent1]/Score[agent2]



