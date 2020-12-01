import numpy as np
import copy
import time
from Timer import timer
from random import random

class AlphaBetaSearch:
    def __init__(self, depth):
        self.depth = depth
        self.policy = {}
        self.totalValue = 0
        self.iter = 0

    def reducedAvailableActions(self, simulator):
        # only search close actions
        timer.startTime("part 1")
        n = simulator.getNumberForWin() // 2
        flag = [[0 for i in range(simulator.boardSize)] for j in range(simulator.boardSize)]
        for action in simulator.getActions():
            for x in range(-n, n + 1, 1):
                for y in range(-n, n + 1, 1):
                    if not simulator.outOfRange(action[0] - x, action[1] - y):
                        flag[action[0] - x][action[1] - y] = 1
        reducedAct = []
        for act in simulator.getAvailableActions():
            if flag[act[0]][act[1]]:
                reducedAct.append(act)
        if len(reducedAct) == 0:  # no stone on board
            reducedAct.append((simulator.getSize() // 2, simulator.getSize() // 2))
        timer.endTime("part 1")

        #timer.startTime("part 2")
        action_importance_pairs = []
        for action in reducedAct:
#            timer.startTime("part 2:calculate importance")
            importance = simulator.updateFeatureAndScore(simulator.getCurrentPlayer(), action, 0) + \
                         simulator.updateFeatureAndScore(simulator.getLastPlayer(), action, 0)
#            timer.endTime("part 2:calculate importance")
            action_importance_pairs.append((importance, action))
        timer.startTime("part 3")
        action_importance_pairs.sort(reverse=True)
        length = len(action_importance_pairs)
        random_pos = [0,3,10]
        for pos in random_pos:
            if length>pos+1 and random()<0.5:
                action_importance_pairs[pos], action_importance_pairs[pos+1] = action_importance_pairs[pos+1], action_importance_pairs[pos]
        timer.endTime("part 3")
        #timer.endTime("part 2")
        return map(lambda pair: pair[1], action_importance_pairs)

    def maxValue(self, alpha, beta, currentDepth, simulator):
        self.iter +=1
        if currentDepth == self.depth:
            return simulator.approxScore()
        value = -1e9
#        timer.startTime("fetch actions")
        availableActions = self.reducedAvailableActions(simulator)
#        timer.endTime("fetch actions")
        for act in availableActions:
            timer.startTime("take action")
            simulator.takeAction(act)
            timer.endTime("take action")
            if simulator.isFinish():
                value = -simulator.approxScore()
                simulator.rollbackLastAction()
                break
            value = max(value, -self.maxValue(-beta, -alpha, currentDepth + 1, simulator))
            timer.startTime("rollback")
            simulator.rollbackLastAction()
            timer.endTime("rollback")
            if value >= beta:
                break
            alpha = max(alpha, value)
        return value

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
        new_simulator = copy.deepcopy(simulator)
        new_simulator.mode = "min-max-search"
        for act in self.reducedAvailableActions(simulator):
            new_simulator.takeAction(act)
            if new_simulator.isFinish():
                value = -new_simulator.approxScore()
            else:
                value = -self.maxValue(-1e9, -best, 0, new_simulator)
            new_simulator.rollbackLastAction()
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
