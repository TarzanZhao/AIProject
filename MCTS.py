

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

    def __init__(self, father, action, prob):
        super(TreeNode, self).__init__()

        self.children = {} # (action, child)

        self.fatherNode = father
        self.action = action

        self.P = prob
        self.N = 0 # visiting times.
        self.V = 0 # average value.
        self.W = 0 # total value.

    def allActions(self):
        return self.children.keys()

    def allChildren(self):
        return self.children.values()

    def PUCT(self, totalN):
        return self.V + 1.0*self.P * (totalN**0.5)/(1+self.N)

    def bestActionByPUCT(self):
        actions = self.allActions()
        totalN = 0
        for action in actions:
            totalN += self.children[action].N

        bestPUCT = -1e9
        bestAction = (-1, -1)
        for action in actions:
            newPUCT = self.children[action].PUCT()
            if newPUCT > bestPUCT:
                bestAction = newPUCT
                bestAction = action
        return action

    def isLeaf(self):
        return len(self.children) == 0





class MCTS:
    def __init__(self, board, eta=1.0, network):
        super(MCTS, self).__init__()
        self.root = TreeNode(None, None, 0)
        self.currentRootNode = self.root

        self.board = board
        self.network = network
        self.eta = eta

    def expand(self):
        """
        reach and expand a leaf.
        :return:
        """
        node = self.currentRootNode
        while not node.isLeaf():
            action = node.bestActionByPUCT()
            node = node.children[action]

        actions = self.board.getAvailableActions()
        actionProbability, z = self.network.getPolicy_Value()
        for action in actions:
            node.children
        pass

    def run(self, numOfIterations):
        for i in range(numOfIterations):
            self.expand()

    def getPolicy(self):
        node = self.currentRootNode
        actions = node.allActions()
        totalN = 0
        for action in actions:
            totalN += node.children[action].N**(1.0/self.eta)
        policy = {}
        for action in actions:
            policy[action] = (node.children[action].N**(1.0/self.eta) ) /totalN
        return policy

