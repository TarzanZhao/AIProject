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
        x = torch.flatten(x,1)

        policy = F.relu(self.policyFc1(x))
        policy = F.softmax(self.policyFc2(policy),dim=1)

        value = F.relu(self.valueFc1(x))
        value = torch.sigmoid(self.valueFc2(value))

        return policy, value

    def getPolicy_Value(self,x):
        x = x.to(self.arg.device)
        x= torch.unsqueeze(x,dim=0)
        x = self.forward(x)
        return torch.squeeze(x[0],dim=0).to(self.arg.device),torch.squeeze(x[1],dim=0).to(self.agr.device)


class MixLoss(nn.Module):
    def __init__(self):
        super(MixLoss, self).__init__()

    def forward(self, predPolicy,labelPolicy, predValue,labelValue):
        valueLoss = F.mse_loss(labelValue,predValue.view_as(labelValue),reduction="mean")
        policyLoss = ((torch.log(predPolicy) * labelPolicy).sum(dim=1)).mean(dim=0)
        return valueLoss - policyLoss
