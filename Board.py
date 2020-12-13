import torch
import copy
import numpy as np
from Timer import timer
import argparse

class Board:
    def __init__(self, boardSize, numberForWin, mode="normal", maxScoreForStep = 10.0, featureUpdate=True):#"min-max-search"
        super(Board, self).__init__()
        self.boardSize = boardSize
        self.numberForWin = numberForWin
        self.mode = mode
        self.featureUpdate = featureUpdate
        self.maxScoreForStep = maxScoreForStep
        self.init()

    def init(self):
        self.currentPlayer = 0  # 0: first player, 1: second player.
        self.actions = []  # (player, (x,y)):
        self.availableActions = []
        for x in range(self.boardSize):
            for y in range(self.boardSize):
                self.availableActions.append((x, y))
        self.boardList = [[[0 for k in range(self.boardSize)] for j in range(self.boardSize)] for i in range(2)]
        self.playerScore = [0, 0]
        self.feature = [[0 for i in range(self.numberForWin+1)] for j in range(2)]
        #record occurrance of max length for each (action, direction) pair.

    def encodeAction(self, action):
        return self.boardSize ** 2 if action == (-1, -1) else action[0] * self.boardSize + action[1]

    def decodeAction(self, code):
        return (-1, -1) if code == self.boardSize ** 2 else (code // self.boardSize, code % self.boardSize)

    def getCurrentPlayer(self):
        """
        :return: 0/1 for current player.
        """
        return self.currentPlayer

    def getLastPlayer(self):
        return self.currentPlayer ^ 1

    def getSize(self):
        return self.boardSize

    def getCurrentState(self):
        """
        :return: a 4*boardSize*boardSize tensor that represents a state for neural network to use. Assume the next(current) player is first.
        """
        boardList = []
        actions = copy.copy(self.actions)
        length = len(actions)
        rollbackactions = []
        for i in range(2):
            boardList.append(self.getBoardList(self.currentPlayer))
            boardList.append(self.getBoardList(self.currentPlayer ^ 1))
            if length > 0:
                rollbackactions.append(actions.pop())
                self.rollbackLastAction()
                length -= 1
            if length > 0:
                rollbackactions.append(actions.pop())
                self.rollbackLastAction()
                length -= 1

        #>> takeAction? return boardList?
        for idx in range(len(rollbackactions)-1, -1, -1):
            action = rollbackactions[idx]
            self.takeAction(action)
        return boardList

    def outOfRange(self, x, y):
        return x < 0 or y < 0 or x >= self.boardSize or y >= self.boardSize

    def updateFeatureAndScore(self, currentPlayer, action, mode):
        dx = [-1, -1, -1, 0]
        dy = [-1, 0, 1, -1]
        x, y = action[0], action[1]
        lengths = []
        board = self.boardList[currentPlayer]
        timer.startTime("part 2:get length")
        importance = 0
        for i in range(4):
            l, r = 0, 0
            xx, yy = x-dx[i], y-dy[i]
            while l+1+1<=self.numberForWin and not self.outOfRange(xx, yy) and board[xx][yy]:
                l += 1
                xx -= dx[i]
                yy -= dy[i]

            xx, yy = x+dx[i], y+dy[i]
            while r+l+1+1<=self.numberForWin and not self.outOfRange(xx, yy) and board[xx][yy]:
                r += 1
                xx += dx[i]
                yy += dy[i]
            lengths.append(l + r + 1)
            importance += self.lenToScore(l+r+1)
        timer.endTime("part 2:get length")

        # timer.startTime("test")
        # tmp = 1
        # for i in range(4):
        #     for j in range(self.numberForWin):
        #         for k in range(self.numberForWin):
        #         tmp += i*j
        # timer.endTime("test")
        if mode == 0:
            return importance

        timer.startTime("part 2:update")
        for length in lengths:
            self.feature[currentPlayer][length] += 1*mode
            if length == self.numberForWin:
                if self.feature[currentPlayer][length] == 0 or self.feature[currentPlayer][length] == 1:
                    self.playerScore[currentPlayer] += self.lenToScore(length) * mode
            else:
                self.playerScore[currentPlayer] += self.lenToScore(length) * mode
        timer.endTime("part 2:update")
        return importance

    def lenToScore(self, length):
        if length == 1:
            return 0
        return self.maxScoreForStep / (20.0**(self.numberForWin-length))

    def approxScore(self):
        return self.playerScore[self.currentPlayer] - self.playerScore[self.currentPlayer^1]

    def takeAction(self, action):
#        print(action)
        if self.featureUpdate:
            self.updateFeatureAndScore(self.currentPlayer, action, mode = 1)

        self.actions.append(action)
        self.availableActions.remove(action)
        self.boardList[self.currentPlayer][action[0]][action[1]]=1
        self.currentPlayer ^= 1

    def rollbackLastAction(self):
        self.currentPlayer ^= 1
        action = self.actions.pop()
#        print("back ", action)
        self.availableActions.append(action)
        self.boardList[self.currentPlayer][action[0]][action[1]]=0

        if self.featureUpdate:
            self.updateFeatureAndScore(self.currentPlayer, action, mode = -1)

    def getBoardList(self, player):
        """
        :param actions: a list of actions.
        :return: a boardSize*boardSize list.
        """
        return self.boardList[player]

    #def getCompleteBoard(self):
    #    return self.getBoardTensor(self.actions, 0) + self.getBoardTensor(self.actions, 1) * 2

    def getAvailableActions(self):
        """
        :return: a list of actions i.e. (x,y)
        """
        return self.availableActions

    def getNumberForWin(self):
        return self.numberForWin

    def getActions(self):
        return self.actions

    def isWin(self, player):
        return self.feature[player][self.numberForWin] > 0

    def isFinish(self):
        b = self.isWin(0) or self.isWin(1)
        return 1 if not b and len(self.actions) == self.boardSize ** 2 else b

    def getWinner(self):
        """
        Assume the winner has been decided.
        :return: 0/1 for winner.
        """
        if self.isWin(0):
            return 0
        elif self.isWin(1):
            return 1
        elif len(self.actions) == self.boardSize ** 2:
            return np.random.randint(0, 2)

    def setFeatureUpdate(self, v):
        self.featureUpdate = v

    def finish(self):
        """
        :return:
        """
        pass
