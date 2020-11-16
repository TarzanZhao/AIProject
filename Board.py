

class Board:
    def __init__(self, boardSize, numberForWin):
        super(Board, self).__init__()
        self.currentPlayer = 0 # 0: first player, 1: second player.
        self.actions = [] # (player, (x,y)):

    def getCurrentPalyer(self):
        """
        :return: 0/1 for current player.
        """
        return self.currentPlayer

    def getCurrentState(self):
        """
        :return: a 4*boardSize*boardSize tensor that represents a state for neural network to use. Assume the next(current) player is first.
        """
        pass

    def takeAction(self, action):
        """
        :param action: (int, int)
        :return:
        """
        pass

    def getAvailableActions(self):
        """
        :return: a list of actions i.e. (x,y)
        """
        pass

    def isWin(self):
        """
        :return: Boolean
        """
        pass

    def getWinner(self):
        """
        :return: 0/1 for winner.
        """
        return self.currentPlayer^1

    def finish(self):
        """
        :return:
        """
        pass
