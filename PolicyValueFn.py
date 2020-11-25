import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.densenet import _DenseBlock, _Transition

import torch.utils.data


class PolicyValueFn(nn.Module):
    def __init__(self,argparse):
        """
        :param argparse: hyper parameter
        """
        super(PolicyValueFn, self).__init__()
        self.arg = argparse
        self.conv = nn.Sequential(
            nn.BatchNorm2d(self.arg.channels),
            nn.Conv2d(self.arg.channels, 32, kernel_size=(1, 1)),
            nn.ReLU(inplace=True)
        )
        self.dense = nn.Sequential(
            _DenseBlock(num_layers=7, num_input_features=32, bn_size=4,
                        growth_rate=12, drop_rate=self.arg.drop_rate),
            _Transition(num_input_features=32 + 7 * 12, num_output_features=12),
            _DenseBlock(num_layers=7, num_input_features=12, bn_size=2,
                        growth_rate=9, drop_rate=self.arg.drop_rate)
        )
        size = (self.arg.size//2)**2*(12+7*9)
        self.policyFc = nn.Linear(size,self.arg.size**2)
        self.valueFc = nn.Linear(size, 1)


    def forward(self,x):
        x = self.conv(x)
        x = self.dense(x)
        x = torch.flatten(x,1)

        policy = F.softmax(self.policyFc(x), dim=1)
        value = torch.tanh(self.valueFc(x))

        return policy, value

    def getPolicy_Value(self,x):
        x = x.to(self.arg.device)
        x= torch.unsqueeze(x,dim=0)
        x = self.forward(x)
        return torch.squeeze(x[0],dim=0).to(self.arg.device),torch.squeeze(x[1],dim=0).to(self.arg.device)


class MixLoss(nn.Module):
    def __init__(self):
        super(MixLoss, self).__init__()

    def forward(self, predPolicy,labelPolicy, predValue,labelValue):
        valueLoss = F.mse_loss(labelValue, predValue.view_as(labelValue),reduction="mean")
        policyLoss = ((torch.log(predPolicy) * labelPolicy).sum(dim=1)).mean(dim=0)
        return valueLoss - policyLoss
