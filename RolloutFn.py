import Agent
import copy
import random

class RolloutFn:
    def __init__(self, cnt = 3):
        self.cnt = cnt

    def __call__(self, simulator):
        return 0

class minMaxRolloutFn(RolloutFn):
    def __init__(self, cnt):
        super(minMaxRolloutFn,self).__init__(cnt)

    def __call__(self, simulator):
        view = simulator.getCurrentPlayer()
        z = 0
        agent = Agent.SearchAgent(depth=0,epsilon=0)
        for i in range(self.cnt):
            simu = copy.deepcopy(simulator)
            while not simu.isFinish():
                action = agent.getAction(simu)
                simu.takeAction(action)
            z+= 1.0 if view == simu.getWinner() else -1.0
        return z/self.cnt

class randomRolloutFn(RolloutFn):
    def __init__(self,cnt):
        super(randomRolloutFn, self).__init__(cnt)

    def __call__(self, simulator):
        view = simulator.getCurrentPlayer()
        z = 0
        for i in range(self.cnt):
            simu = copy.deepcopy(simulator)
            while not simu.isFinish():
                action = random.choice(simu.getAvailableActions())
                simu.takeAction(action)
            z+= 1.0 if view == simu.getWinner() else -1.0
        return z/self.cnt

