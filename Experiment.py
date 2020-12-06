import Agent
import Board
import Game
import matplotlib.pyplot as plt
from DataStorage import dataProcessor


class Experiment():
    def __init__(self, args=None):
        self.args = args

    def getWinScore(self, game, agent):
        win = 0
        for i in range(self.args.numOfEvaluations):
            if game.run() == agent:
                win += 1
        return win / self.args.numOfEvaluations

    def selfplayWithDifferentNumOfIterations(self, modelID=None):
        model = dataProcessor.loadNetwork(self.args, modelID)
        winRate = []
        for iter in range(5, 50, 5):
            agent = Agent.IntelligentAgent(iter, model)
            board = Board.Board(self.args.size, self.args.numberForWin)
            game = Game.Game(agent, agent, board)
            winRate.append(self.getWinScore(game, agent))
            print("Iterations: %d, WinRate: %.2f" % (iter, winRate[-1]))

        plt.figure()
        plt.xlabel("Number of Iterations per Movem")
        plt.ylabel("Win Rate for First Player")
        plt.title("The must-win-strategy in 6*6 board with 3 in a row is learnable")
        plt.plot(range(5, 50, 5), winRate, color='blue', linestyle='-')
        plt.savefig('figure/6-6 3 must-win-strategy.png')
        plt.show()
