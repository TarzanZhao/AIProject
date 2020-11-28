import torch
import numpy as np
from DataStorage import dataProcessor
import GeneralSearch
import MCTS
from Timer import timer

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
    def __init__(self, numOfiterations, network, path):
        self.datalist = []
        self.numOfiterations = numOfiterations
        self.network = network
        self.mcts = MCTS.MCTS()
        self.path = path
        self.finalDataList = []
        self.isFinished = 0
        pass

    def init(self):
        self.datalist = []
        self.mcts = MCTS.MCTS()
        self.isFinished = 0

    def getAction(self, simulator):
        TimeID = timer.startTime("get action from selfPlayAgent")
        self.mcts.run(self.numOfiterations, simulator, self.network)
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
        if not self.isFinished:
            self.isFinished = 1
            for i in self.datalist:
                z = 1.0 if i[2]==winner else -1.0
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
    def __init__(self, numOfiterations, network):
        self.numOfiterations = numOfiterations
        self.network = network
        self.act_pro_pair = {}

    def getAction(self, simulator):
        mcts = MCTS.MCTS(C=5)
        mcts.run(self.numOfiterations,simulator,self.network)
        self.act_pro_pair = mcts.getPolicy()
        p = 0
        action = (-1,-1)
        for (act,pro) in self.act_pro_pair.items():
            if pro>p:
                p = pro
                action = act
        return action

    def getActionProPair(self):
        return self.act_pro_pair

    def __str__(self):
        return "IntelligentAgent Instance"

class SearchAgent(Agent):
    def __init__(self, depth=6):
        self.depth = depth
        self.tree = GeneralSearch.AlphaBetaSearch(self.depth)

    def getAction(self, simulator):
        return self.tree.getAction(simulator)

    def getActionProPair(self):
        return self.tree.getPolicy()

    def __str__(self):
        return "SearchAgent Instance"