import torch
import numpy as np
from DataStorage import dataProcessor
import GeneralSearch
import MCTS
from Timer import timer
from random import random
from PolicyValueFn import ExpandingFn
from RolloutFn import randomRolloutFn, minMaxRolloutFn

class Agent:
    def getAction(self, simulator):
        """
        :param simulator: a game state simulator (can do query)
        :return: an available action (x,y)
        """
        pass

    def finish(self, isWin):
        """
        :param isWin: is the Agent win
        """
        pass

    def getActionProPair(self):
        """
        :return: stochastic policy
        """
        pass

    def init(self):
        pass


class SelfplayAgent(Agent):
    def __init__(self, numOfiterations, network, path, eta = 1.0, decay = 0.85, rollout=None, balance = 0):
        self.datalist = []
        self.numOfiterations = numOfiterations
        self.network = ExpandingFn(network)
        self.eta = eta
        self.decay = decay
        self.balance = balance
        self.mcts = MCTS.MCTS(eta= self.eta)
        self.path = path
        self.finalDataList = []
        self.isFinished = 0
        self.rollout = rollout
        pass

    def init(self):
        self.datalist = []
        self.mcts = MCTS.MCTS(eta= self.eta)
        self.isFinished = 0

    def getAction(self, simulator):
        TimeID = timer.startTime("Get action")
        self.mcts.run(self.numOfiterations, simulator, self.network, rolloutFn=self.rollout, balance=self.balance)
        act_pro_pair = self.mcts.getPolicy()
        keys = []
        values = []
        for key, value in act_pro_pair.items():
            keys.append(key)
            values.append(value)
        action = keys[np.random.choice(len(values), 1, p=values)[0]]
        self.mcts.takeAction(action)
        policy = [0 for i in range(simulator.getSize()**2)]
        for act in act_pro_pair.keys():
            policy[simulator.encodeAction(act)] = act_pro_pair[act]
        self.datalist.append((action, policy, simulator.getCurrentPlayer()))
        timer.endTime(TimeID)
        return action

    def finish(self, winner):
        gamma = self.decay*(len(self.datalist)-1)
        if not self.isFinished:
            self.isFinished = 1
            for i in self.datalist:
                z = gamma *1.0 if i[2]==winner else gamma*(-1.0)
                gamma /= self.decay
                self.finalDataList.append((i[0], i[1], z))
            self.finalDataList.append('end')

    def saveData(self):
        dataProcessor.saveData(self.finalDataList, self.path)

    def __str__(self):
        return "SelfplayAgent Instance"


class RandomAgent(Agent):
    def __init__(self):
        pass

    def getAction(self, simulator):
        list = simulator.getAvailableActions()
        return list[np.random.randint(0, len(list))]

    def __str__(self):
        return "RandomAgent Instance"


class IntelligentAgent(Agent):
    def __init__(self, numOfiterations, network, rolloutFn=None , balance = 0):
        self.numOfiterations = numOfiterations
        self.network = ExpandingFn(network)
        self.act_pro_pair = {}
        self.rollout = rolloutFn
        self.balance = balance

    def getAction(self, simulator):
        mcts = MCTS.MCTS(C=5)
        mcts.run(self.numOfiterations,simulator,self.network, rolloutFn=self.rollout, balance=self.balance)
        self.act_pro_pair = mcts.getPolicy()
        p = 0
        action = (-1,-1)
        for (act,pro) in self.act_pro_pair.items():
            if pro>p:
                p = pro
                action = act
            elif pro==p and np.random.random()>0.5:
                action = act
        return action

    def getActionProPair(self):
        return self.act_pro_pair

    def setNumOfIterations(self, numOfIterations):
        self.numOfiterations = numOfIterations

    def __str__(self):
        return "IntelligentAgent Instance"

class NetWorkAgent(Agent):
    def __init__(self, network):
        self.net = network
        self.actProPair = {}

    def getAction(self, simulator):
        TimeId = timer.startTime("GetAction")
        self.policy,_ = self.net.getPolicy_Value(torch.tensor(simulator.getCurrentState(), dtype=torch.float))
        self.policy = self.policy.tolist()
        bestAction = (-1, -1)
        bestProb = -1e9
        self.actProPair = {}
        for act in simulator.getAvailableActions():
            prob = self.policy[simulator.encodeAction(act)]
            self.actProPair[act] = prob
            if prob > bestProb:
                bestAction = act
                bestProb = prob
            elif prob == bestProb and np.random.random() > 0.5:
                bestAction = act
        timer.endTime(TimeId)
        return bestAction

    def getActionProPair(self):
        return self.actProPair

class SearchAgent(Agent):
    def __init__(self, depth=6, epsilon = 0.3):
        self.depth = depth
        self.tree = GeneralSearch.AlphaBetaSearch(self.depth)
        self.epsilon = epsilon
        self.dataList = []

    def init(self):
        self.dataList = []

    def getAction(self, simulator):
        bestAction = self.tree.getAction(simulator)
        act_pro_pair = self.tree.getPolicy()
        keys = []
        values = []
        for key, value in act_pro_pair.items():
            keys.append(key)
            values.append(value)
#        print(values)
        action = keys[np.random.choice(len(values), p=values)]
        if random() < self.epsilon: #(1.0-2*self.depth/10.0)
            finalAction = action
        else:
            finalAction = bestAction

        policy = [0 for i in range(simulator.getSize() ** 2)]
        for act, prob in zip(keys, values):
            policy[simulator.encodeAction(act)] = prob
        self.dataList.append((finalAction, policy))
        return finalAction

    def getActionProPair(self):
        return self.tree.getPolicy()

    def __str__(self):
        return "SearchAgent Instance"

class GreedyAgent(SearchAgent):
    def __init__(self):
        super().__init__(depth=0)
        #print(self.depth)

    def __dir__(self):
        return "GreedyAgent Instance"

    # def getAction(self, simulator):
    #     return self.tree.getAction(simulator)
    #     #self.policy = self.tree.reducedAvailableActions(simulator, returnPolicy = False)
    #
    # def getActionProPair(self):
    #     return self.policy

