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
            if action in self.simulator.getAvailableActions():
                self.simulator.takeAction(action)

        winner = self.simulator.getWinner()
        self.agent0.finish(winner == 0)
        self.agent1.finish(winner == 1)
        self.simulator.finish()

        return agentMap[winner]

    def __str__(self):
        return "Game Instance"

