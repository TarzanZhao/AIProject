import MCTS
import torch
import Board
import numpy
import DataStorage

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

    def init(self):
        pass



class SelfplayAgent(Agent):
    def __init__(self, numofIterations,network,path):
        self.datalist =[]
        self.numofIterations = numofIterations
        self.network = network
        self.mcts = MCTS.MCTS()
        self.path = path
        pass

    def init(self):
        self.datalist = []
        self.mcts = MCTS.MCTS()

    def getAction(self, simulator):
        self.mcts.run(self.numofIterations,simulator,self.network)
        act_pro_pair = self.mcts.getPolicy()

        policy = numpy.zeros(simulator.getSize() ** 2)
        for act in act_pro_pair.keys():
            policy[simulator.encodeAction(act)]=act_pro_pair[act]
        self.datalist.append((simulator.getCurrentState(), torch.tensor(policy)))

        return max(act_pro_pair.items(), key=lambda act_pro: act_pro[1])[0]


    def finish(self, isWin):
        if isWin:
            self.finalDataList = []
            z = isWin = 1
            for i in range(len(self.datalist)-1,-1,-1):
                self.finalDataList.append((self.datalist[i][0],self.datalist[i][1],z))
                z = 1 - z
            DataStorage.saveData(self.datalist, self.path)

    def __str__(self):
        return "SelfplayAgent Instance"


class RandomAgent(Agent):
    def __init__(self):
        pass

    def __str__(self):
        return "RandomAgent Instance"

class HumanAnent(Agent):
    def __init__(self):
        pass

    def __str__(self):
        return "HumanAgent Instance"

class IntelligentAgent(Agent):
    def __init__(self):
        pass

    def __str__(self):
        return "IntelligentAgent Instance"