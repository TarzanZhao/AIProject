import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

import torch.utils.data


class PolicyValueFn(nn.Module):
    def __init__(self,argparse):
        """
        :param argparse: hyper parameter
        """
        super(PolicyValueFn, self).__init__()
        self.arg = argparse
        self.conv1 = nn.Conv2d(self.arg.channels,64,kernel_size=(3,3),padding=1)
        self.conv2 = nn.Conv2d(64,128,kernel_size=(3,3),padding=1)
        self.policyFc1 = nn.Linear(argparse.size**2*128,1000)
        self.policyFc2 = nn.Linear(1000,argparse.size**2+1)
        self.valueFc1 = nn.Linear(argparse.size**2*128,100)
        self.valueFc2 = nn.Linear(100,1)


    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x,dims=1)

        policy = F.relu(self.policyFc1(x))
        policy = F.log_softmax(self.policyFc2(policy))

        value = F.relu(self.valueFc1(x))
        value = F.softmax(self.valueFc2(value))
        return policy, value

    def getPolicy_Value(self,state):
        with torch.no_grad():
            policy, value = self.forward(state)




class MixLoss(nn.Module):
    def __init__(self):
        super(MixLoss, self).__init__()

    def forward(self, log_predPolicy,labelPolicy, predValue,labelValue):
        valueLoss = F.mse_loss(labelValue,predValue)
        policyLoss =
        return valueLoss - policyLoss


class SelfPlayDataset(torch.utils.data.Dataset):
    def __init__(self,path):
        file = open(path)

        self.dataList = [] # need to continue


    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, item):
        data = self.dataList[item]
        state = torch.tensor(data[0],dtype=torch.float)
        labelPolicy = torch.tensor(data[1],dtype=torch.float)
        labelValue = torch.tensor(data[2],dtype=torch.float)

