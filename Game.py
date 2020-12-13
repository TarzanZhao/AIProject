from Timer import timer
import torch
import argument
import logger

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
        self.args = argument.get_args()
        self.logger = logger.get_logger()
        self.num_run = 0

    def gameInit(self):
        self.agent0.init()
        self.agent1.init()
        self.simulator.init()

    def switchAgents(self):
        self.agent0, self.agent1 = self.agent1, self.agent0

    def run(self):
        """
        :return: winner of the game
        """
        self.gameInit()

        agentMap = {0: self.agent0, 1: self.agent1}
        while not self.simulator.isFinish():
            agent = agentMap[self.simulator.getCurrentPlayer()]
            action = agent.getAction(self.simulator)
#            print(f"player {self.simulator.getLastPlayer()} take action {action}")
            # print("---player %d's round, take action (%d,%d) "% (self.simulator.currentPlayer, action[0], action[1]))
            #            print(bd.numpy().tolist())
            if action in self.simulator.getAvailableActions():
                self.simulator.takeAction(action)
#            print(action)
            # bd = self.simulator.getCompleteBoard().numpy().tolist()
            # for i in range(self.simulator.getSize()):
            #     for j in range(self.simulator.getSize()):
            #         print(int(bd[i][j]), end=" ")
            #     print("")

        winner = self.simulator.getWinner()
        if self.args is not None and self.args.todo == 'sampledata':
            if self.num_run % self.args.n_log_step == 0:
                self.logger.info(torch.Tensor(self.simulator.getBoardList(0))+torch.Tensor(self.simulator.getBoardList(1))*2)
                self.logger.info("Num of play: %d" % len(self.simulator.actions))
            self.num_run += 1


#        print(self.simulator.actions)
        self.agent0.finish(winner)
        self.agent1.finish(winner)
        self.simulator.finish()
        return agentMap[winner]

    def __str__(self):
        return "Game Instance"

