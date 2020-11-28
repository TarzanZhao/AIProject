from Timer import timer
class Game:
    def __init__(self, agent0, agent1, simulator):
        """
        :param agent0: first player
        :param agent1: second player
        :param simulator: game simulator (can query)
        """
        self.agent0 = agent0
        self.agent1 = agent1
        self.simulator = simulator

    def gameInit(self):
        self.agent0.init()
        self.agent1.init()
        self.simulator.init()

    def run(self):
        """
        :return: winner of the game
        """
        self.gameInit()

        agentMap = {0: self.agent0, 1: self.agent1}
        while not self.simulator.isFinish():
            agent = agentMap[self.simulator.getCurrentPlayer()]
            action = agent.getAction(self.simulator)
            # print("---player %d's round, take action (%d,%d) "% (self.simulator.currentPlayer, action[0], action[1]))
            #            print(bd.numpy().tolist())
            if action in self.simulator.getAvailableActions():
                self.simulator.takeAction(action)
            # bd = self.simulator.getCompleteBoard().numpy().tolist()
            # for i in range(self.simulator.getSize()):
            #     for j in range(self.simulator.getSize()):
            #         print(int(bd[i][j]), end=" ")
            #     print("")

        winner = self.simulator.getWinner()
        print("Num of play: %d" %len(self.simulator.actions))
        self.agent0.finish(winner)
        self.agent1.finish(winner)
        self.simulator.finish()
        return agentMap[winner]

    def __str__(self):
        return "Game Instance"

