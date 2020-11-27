import copy
import numpy as np
from Timer import timer
# class TreeEdge:
#     def __init__(self, fatherNode, childNode, action):
#         super(TreeEdge, self).__init__()
#         self.fatherNode = fatherNode
#         self.childNode = childNode
#         self.action = action
#
#         self.N = 0
#         self.Q =

class TreeNode:
    def __init__(self, father, action, prob, C=5):
        super(TreeNode, self).__init__()

        self.children = {} # (action, child)

        self.fatherNode = father
        self.action = action

        self.P = prob
        self.N = 0 # visiting times.
        self.V = 0 # average value.
        self.W = 0 # total value.
        self.C = C

    def allActions(self):
        return self.children.keys()

    def allChildren(self):
        return self.children.values()

    def PUCT(self, totalN):
        return self.V + self.C*self.P * (totalN**0.5)/(self.N+1)

    def bestActionByPUCT(self):
        actions = self.allActions()
        totalN = 0
        for action in actions:
            totalN += self.children[action].N

        bestPUCT = -1e9
        bestAction = []
        for action in actions:
            newPUCT = self.children[action].PUCT(totalN=totalN)
            if newPUCT > bestPUCT:
                bestPUCT = newPUCT
                bestAction = []
                bestAction.append(action)
            elif newPUCT == bestPUCT:
                bestAction.append(action)
        return bestAction[np.random.randint(len(bestAction))]

    def isLeaf(self):
        return len(self.children) == 0

    def __del__(self):
        del self.children



class MCTS:
    def __init__(self, eta=1.0, C = 5):
        super(MCTS, self).__init__()
        self.currentRootNode = TreeNode(None, None, 0, C)
        self.eta = eta
        self.C = C

    def expand(self, simulator, network):
        """
        reach and expand a leaf.
        :return:
        """
        Selection = timer.startTime("MCTS selection")
        simulator = copy.deepcopy(simulator) #could I use copy?
        node = self.currentRootNode
        while not node.isLeaf():
            Child = timer.startTime("MCTS best child")
            action = node.bestActionByPUCT()
            timer.endTime(Child)
            node = node.children[action]
            simulator.takeAction(action)
        timer.endTime(Selection)
        if simulator.isFinish():
            z = 1.0 if simulator.getWinner() == simulator.getCurrentPlayer() else -1.0
        else:
            actions = simulator.getAvailableActions()
            networkCall = timer.startTime("calling network")
            network.eval()
            actionProbability, z = network.getPolicy_Value(simulator.getCurrentState())
            timer.endTime(networkCall)
            Adding = timer.startTime("MCTS adding nodes")
            for action in actions:
                node.children[action] = TreeNode(node, action, actionProbability[simulator.encodeAction(action)],self.C)
            timer.endTime(Adding)
        BackPro = timer.startTime("MCTS backpropagation")
        while node != self.currentRootNode:
            node.N += 1
            node.W += -z  #in logic, 一个点的Q存的是他父亲走这一步的价值
            node.V = node.W/node.N
            z=-z
            node = node.fatherNode
        timer.endTime(BackPro)

    def run(self, numOfIterations, simulator, network):
        for i in range(numOfIterations):
            #print("------iter %d"%i)
            self.expand(simulator, network)

    def getPolicy(self):
        TimeID = timer.startTime("MCTS get policy")
        node = self.currentRootNode
        actions = node.allActions()
        totalN = 0
        for action in actions:
            totalN += node.children[action].N**(1.0/self.eta)
        policy = {}
        for action in actions:
            policy[action] = (node.children[action].N**(1.0/self.eta))/totalN
        timer.endTime(TimeID)
        return policy

    def takeAction(self,action):
        keys = list(self.currentRootNode.children.keys())
        for act in keys:
            if act != action:
                del self.currentRootNode.children[act]
        self.currentRootNode = self.currentRootNode.children[action]