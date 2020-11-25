import copy
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
        bestAction = (-1, -1)
        for action in actions:
            newPUCT = self.children[action].PUCT(totalN=totalN)
            if newPUCT > bestPUCT:
                bestPUCT = newPUCT
                bestAction = action
        return bestAction

    def isLeaf(self):
        return len(self.children) == 0





class MCTS:
    def __init__(self, eta=1.0):
        super(MCTS, self).__init__()
        self.root = TreeNode(None, None, 0)
        self.currentRootNode = self.root
        self.eta = eta

    def expand(self, simulator, network):
        """
        reach and expand a leaf.
        :return:
        """
        simulator = copy.deepcopy(simulator) #could I use copy?
        node = self.currentRootNode
        while not node.isLeaf():
            action = node.bestActionByPUCT()
            node = node.children[action]
            simulator.takeAction(action)

        if simulator.isFinish():
            z = 1.0 if simulator.getWinner() == simulator.getCurrentPlayer() else -1.0
        else:
            actions = simulator.getAvailableActions()
            network.eval()
            actionProbability, z = network.getPolicy_Value(simulator.getCurrentState())
            for action in actions:
                node.children[action] = TreeNode(node, action, actionProbability[simulator.encodeAction(action)])
        while node != self.currentRootNode:
            node.N += 1
            node.W += -z  #in logic, 一个点的Q存的是他父亲走这一步的价值
            node.V = node.W/node.N
            z=-z
            node = node.fatherNode


    def run(self, numOfIterations, simulator, network):
        for i in range(numOfIterations):
            #print("------iter %d"%i)
            self.expand(simulator, network)

    def getPolicy(self):
        node = self.currentRootNode
        actions = node.allActions()
        totalN = 0
        for action in actions:
            totalN += node.children[action].N**(1.0/self.eta)
        policy = {}
        for action in actions:
            policy[action] = (node.children[action].N**(1.0/self.eta))/totalN
        return policy

    def takeAction(self,action):
        self.currentRootNode = self.currentRootNode.children[action]