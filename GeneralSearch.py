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
#        timer.startTime("part 1")
        n = simulator.getNumberForWin() // 2
        flag = [[0 for i in range(simulator.boardSize)] for j in range(simulator.boardSize)]
        dis = [[0 for i in range(simulator.boardSize)] for j in range(simulator.boardSize)]
        actions = simulator.getActions()
        if not actions:
            return [(simulator.getSize() // 2, simulator.getSize() // 2)]

        player = -1

        Q = []
        for action in actions:
            flag[action[0]][action[1]] = player
            player = player^1
            Q.append(action)

        dx = [-1, -1, -1, 0, 1, 1, 1, 0]
        dy = [-1, 0, 1, -1, 1, 0, -1, 1]
        p2 = [0, 3, 6, 9, 12, 15, 18, 21]
        mark = 7

        H = 0
        nActions = len(Q)
        while H<len(Q):
            point = Q[H]
            H+=1
            point_dis = max(flag[point[0]][point[1]], 0)
            for i in range(8):
                xx, yy = point[0]+dx[i], point[1]+dy[i]
                if not simulator.outOfRange(xx, yy) and flag[xx][yy]==0:
                    flag[xx][yy] = point_dis+1
                    if flag[xx][yy]<= n:
                        Q.append((xx,yy))
#        print(actions)
        sorted_actions = sorted(actions)
        for action in sorted_actions:
            x, y = action
            for i in range(4):
                xx, yy = x+dx[i], y+dy[i]
                if not simulator.outOfRange(xx, yy) and flag[xx][yy] == flag[x][y]:
                    dis[x][y] += ((dis[xx][yy] >> p2[i] & mark) + 1) << p2[i]
                else:
                    dis[x][y] += 1<<p2[i]

#        print(type(sorted_actions))
#        sorted_actions
        for j in range(len(sorted_actions)-1, -1, -1):
            x, y = sorted_actions[j]
            for i in range(4,8):
                xx, yy = x + dx[i], y + dy[i]
                if not simulator.outOfRange(xx, yy) and flag[xx][yy] == flag[x][y]:
                    dis[x][y] += ((dis[xx][yy] >> p2[i] & mark) + 1) << p2[i]
                else:
                    dis[x][y] += 1 << p2[i]

        action_importance_pairs = []
        for j in range(nActions, len(Q)+1):

            if j == len(Q) or flag[Q[j][0]][Q[j][1]]!=1:
                action_importance_pairs.sort(reverse=True)
                length = len(action_importance_pairs)
                random_pos = [0, 3, 10]
                for pos in random_pos:
                    if length > pos + 1 and random() < 0.5:
                        action_importance_pairs[pos], action_importance_pairs[pos + 1] = \
                            action_importance_pairs[pos + 1], action_importance_pairs[pos]
                final_sorted_actions = list(map(lambda x:x[1], action_importance_pairs))
                final_sorted_actions.extend(Q[j:])
                return final_sorted_actions

            x, y = Q[j]
            importance = 0
            for i in range(4):
                l = [0,0]
                xx, yy = x+dx[i], y+dy[i]
                if not simulator.outOfRange(xx, yy) and flag[xx][yy]<0:
                    l[flag[xx][yy]+2] += (dis[xx][yy]>>p2[i] & mark)
                xx, yy = x+dx[i+4], y+dy[i+4]
                if not simulator.outOfRange(xx, yy) and flag[xx][yy]<0:
                    l[flag[xx][yy]+2] += (dis[xx][yy]>>p2[i+4] & mark)

                if l[0]!=0 and l[1]!=0:
                    importance += max(simulator.lenToScore(l[0]+1)+simulator.lenToScore(l[1]), simulator.lenToScore(l[0])+simulator.lenToScore(l[1]+1))
                elif l[0]!=0:
                    importance += simulator.lenToScore(l[0] + 1)
                elif l[1]!=0:
                    importance += simulator.lenToScore(l[1] + 1)

            action_importance_pairs.append((importance, (x, y)))

        raise NotImplementedError

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
