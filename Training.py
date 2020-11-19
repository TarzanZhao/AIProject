from DataStorage import retrieveData
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimfrom torch.utils.data import Dataset, DataLoader
from PolicyValueFn import PolicyValueFn
from PolicyValueFn import MixLoss

class MyDataset(Dataset):
    def __init__(self, dataList):
        self.len = len(dataList)
        self.dataList = dataList

    def __getitem__(self, index):
        return self.dataList[index][0], self.dataList[index][1], self.dataList[index][2]

    def __len__(self):
        return self.len

import os


class Training:
    def __init__(self):
        pass
    def train(self, file, numOfEpoch, iteration):
        dataList = retrieveData(file)
        trainSet = MyDataset(dataList)
        dataloader = DataLoader(dataset = trainSet, 
                        batch_size = 128, 
                        num_workers=4, 
                        pin_memory=True,
                        shuffle=True
                        )
        criterion = MixLoss()

        network = PolicyValueFn(argparse)
        if iteration>0:
            network.load_state_dict(torch.load(f'network-{iteration-1}.pt'))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        network.to(device)
        optimizer = optim.SGD(
            network.parameters(), lr=0.001
        )
        trainLossFile="./trainingloss-"+str(iteration)
        if os.path.exists(trainLossFile):
            os.remove(trainLossFile)
            print("successfully deleted "+trainLossFile)

        for epoch in range(numOfEpoch):
            total_loss = 0
            loss = 0
            cnt = 0
            for i, batch in enumerate(dataloader):
                x, y1, y2 = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                z1, z2 = network(x)
                loss = criterion(z1, y1, z2, y2)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cnt += 1
                if cnt%100==0:
                    with open(trainLossFile, mode="a") as file:
                        file.write("i={} loss={}\n".format(i, loss.item()))
            total_loss = total_loss/cnt
            print("epoch={} average_loss={}\n".format(epoch, total_loss))
            with open(trainLossFile, mode="a") as file:
                file.write("epoch={} average_loss={}\n".format(epoch, loss))

            if epoch % 10 == 9:
                torch.save(network.state_dict(), f"./network-{iteration}.pth")

