import numpy as np
import copy
import time
from Timer import timer


class AlphaBetaSearch:
    def __init__(self, depth):
        self.depth = depth
        self.policy = {}
        self.totalValue = 0
        self.iter = 0

    def reducedAvailableActions(self, simulator):
        # only search close actions
        n = simulator.getNumberForWin() // 2
        flag = {}
        for action in simulator.getActions():
            for x in range(-n, n + 1, 1):
                for y in range(-n, n + 1, 1):
                    flag[(action[0] - x, action[1] - y)] = 1
        reducedAct = []
        for act in simulator.getAvailableActions():
            if act in flag:
                reducedAct.append(act)
        if len(reducedAct) == 0:  # no stone on board
            reducedAct.append((simulator.getSize() // 2, simulator.getSize() // 2))
        np.random.shuffle(reducedAct)
        return reducedAct

    def maxValue(self, alpha, beta, currentDepth, simulator):
        self.iter +=1
        if currentDepth == self.depth:
            return self.scoreFn(simulator)
        value = -1e9
        for act in self.reducedAvailableActions(simulator):
            new_simulator = copy.deepcopy(simulator)
            new_simulator.takeAction(act)
            if new_simulator.isFinish():
                return 100
            value = max(value, -self.maxValue(-beta, -alpha, currentDepth + 1, new_simulator))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    # def expectedValue(self,currentDepth):
    def oneSideScore(self, simulator, player):
        board = simulator.getBoardTensor(simulator.getActions(), player)
        dx = [-1, -1, -1, 0]
        dy = [-1, 0, 1, -1]

        def outOfRange(x, y):
            return x < 0 or y < 0 or x >= simulator.getSize() or y >= simulator.getSize()

        feature = {simulator.getNumberForWin() - 2: 0, simulator.getNumberForWin() - 1: 0,
                   simulator.getNumberForWin(): 0}
        for x in range(simulator.getSize()):
            for y in range(simulator.getSize()):
                if board[x][y] == 1:
                    for i in range(4):
                        for j in range(1, simulator.getNumberForWin() + 1):
                            if outOfRange(x + dx[i] * j, y + dy[i] * j) or board[x + dx[i] * j][
                                y + dy[i] * j] == 0:
                                if j in feature:
                                    feature[j] += 1
                                break
        score = 1 if feature[simulator.getNumberForWin()] > 0 else 0
        score += 0.3 if feature[simulator.getNumberForWin() - 1] > 0 else 0
        score += 0.2 if feature[simulator.getNumberForWin() - 2] > 1 else 0
        score += 0.05 if feature[simulator.getNumberForWin() - 2] > 0 else 0
        return score

    def scoreFn(self, simulator):
        return self.oneSideScore(simulator, simulator.getCurrentPlayer()) \
               - self.oneSideScore(simulator, simulator.getLastPlayer())

    def getAction(self, simulator):
        best = -1e9
        bestAction = (-1, -1)
        self.policy = {}
        self.totalValue = 0
        for act in simulator.getAvailableActions():
            self.policy[act] = 0
        for act in self.reducedAvailableActions(simulator):
            new_simulator = copy.deepcopy(simulator)
            new_simulator.takeAction(act)
            value = -self.maxValue(-1e9, -best, 0, new_simulator)
            self.totalValue += np.exp(value)
            self.policy[act] = np.exp(value)
            if value > best:
                best = value
                bestAction = act
        return bestAction

    def getPolicy(self):
        print("Search Iter: %d" %self.iter)
        self.iter = 0
        for key in self.policy.keys():
            self.policy[key] /= self.totalValue
        return self.policy
