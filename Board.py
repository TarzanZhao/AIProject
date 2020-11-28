import torch
import copy
import numpy as np
from Timer import timer


class Board:
    def __init__(self, boardSize, numberForWin):
        super(Board, self).__init__()
        self.currentPlayer = 0  # 0: first player, 1: second player.
        self.actions = []  # (player, (x,y)):
        self.boardSize = boardSize
        self.numberForWin = numberForWin
        self.availableActions = []
        for x in range(self.boardSize):
            for y in range(self.boardSize):
                self.availableActions.append((x, y))
        self.boardList = [[[0 for k in range(boardSize)] for j in range(boardSize)] for i in range(2)]

    def init(self):
        self.currentPlayer = 0  # 0: first player, 1: second player.
        self.actions = []  # (player, (x,y)):
        self.availableActions = []
        for x in range(self.boardSize):
            for y in range(self.boardSize):
                self.availableActions.append((x, y))
        self.boardList = [[[0 for k in range(self.boardSize)] for j in range(self.boardSize)] for i in range(2)]

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
        #boardList = []
        #actions = copy.copy(self.actions)
        #length = len(actions)
        #for i in range(2):
            #boardList.append(self.getBoardTensor(actions, self.currentPlayer))
            #boardList.append(self.getBoardTensor(actions, self.currentPlayer ^ 1))
            #if length > 0:
            #    actions.pop()
            #    length -= 1
            #if length > 0:
            #    actions.pop()
            #    length -= 1
        #return torch.stack(boardList, 0)
        return self.boardList

    def takeAction(self, action):
        """
        :param action: (int, int)
        :return:
        """
        self.actions.append(action)
        self.boardList[self.currentPlayer][action[0]][action[1]]=1
        self.currentPlayer ^= 1
        self.availableActions.remove(action)

    def rollbackLastAction(self):
        act = self.actions[-1]
        self.actions.pop()
        self.availableActions.append(act)
        self.currentPlayer ^=1
        self.boardList[self.currentPlayer][act[0]][act[1]]=0

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
        """
        :return: Boolean
        """
        Win = timer.startTime("isWin")
        board = self.getBoardList(player)
        dx = [-1, -1, -1, 0]
        dy = [-1, 0, 1, -1]

        def outOfRange(x, y):
            return x < 0 or y < 0 or x >= self.boardSize or y >= self.boardSize

        for x in range(self.boardSize):
            for y in range(self.boardSize):
                if board[x][y] == 1:
                    for i in range(4):
                        test = True
                        for j in range(self.numberForWin):
                            if outOfRange(x + dx[i] * j, y + dy[i] * j) or board[x + dx[i] * j][y + dy[i] * j] == 0:
                                test = False
                                break
                        if test:
                            return True

        timer.endTime(Win)
        return False

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

    def finish(self):
        """
        :return:
        """
        pass
