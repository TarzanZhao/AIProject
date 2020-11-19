from DataStorage import retrieveData
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PolicyValueFn import PolicyValueFn
from PolicyValueFn import MixLoss
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
                    m = torch.flip(m, dim=-1)
                m = torch.rot90(m, k=rotate)

                mp = np.array(size**2)
                for x in range(size):
                    for y in range(size):

                        v = m[x][y]
                        mp[v] = x*size+y
                self.map.append(mp)



    def __getitem__(self, index):
        id = index//8
        trans = index%8
        
        Y = torch.zeros(self.size**2)
        for i, v in enumerate(self.dataList[id][1]):
            Y[ self.map[trans][i] ] = v
        
        X = self.dataList[id][0].clone()
        if trans>=4:
            X = torch.flip(X, dim=-1)
            trans-=4
        X = torch.rot90(X, k=trans)
                
        return X, Y, self.dataList[id][2]

    def __len__(self):
        return self.len*8


import os


class Training:
    def __init__(self,arg):
        self.arg = arg
        pass

    def train(self, file, numOfEpoch, currentModel):
        dataList = retrieveData(file)
        trainSet = MyDataset(dataList)
        dataloader = DataLoader(dataset=trainSet,
                                batch_size=128,
                                num_workers=4,
                                pin_memory=True,
                                shuffle=True
                                )
        criterion = MixLoss()

        network = PolicyValueFn(self.arg)
        if currentModel > 0:
            network.load_state_dict(torch.load(f'network-{currentModel - 1}.pt'))

        network.to(self.arg.device)
        optimizer = optim.SGD(
            network.parameters(), self.arg.lr
        )
        trainLossFile = "./trainingloss-" + str(currentModel)
        if os.path.exists(trainLossFile):
            os.remove(trainLossFile)
            print("successfully deleted " + trainLossFile)

        for epoch in range(numOfEpoch):
            total_loss = 0
            loss = 0
            cnt = 0
            for i, batch in enumerate(dataloader):
                x, y1, y2 = batch[0].to(self.arg.device), batch[1].to(self.arg.device), batch[2].to(self.arg.device)
                z1, z2 = network(x)
                loss = criterion(z1, y1, z2, y2)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cnt += 1
                if cnt % 100 == 0:
                    with open(trainLossFile, mode="a") as file:
                        file.write("i={} loss={}\n".format(i, loss.item()))
            total_loss = total_loss / cnt
            print("epoch={} average_loss={}\n".format(epoch, total_loss))
            with open(trainLossFile, mode="a") as file:
                file.write("epoch={} average_loss={}\n".format(epoch, loss))

            if epoch % 10 == 9:
                torch.save(network.state_dict(), f"./network-{currentModel}.pth")
