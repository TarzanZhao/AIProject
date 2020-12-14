import Agent
import copy
import random

def minMaxRolloutFn(simulator):
    view = simulator.getCurrentPlayer()
    z = 0
    cnt = 1
    agent = Agent.SearchAgent(depth=0,epsilon=0)
    for i in range(cnt):
        simu = copy.deepcopy(simulator)
        while not simu.isFinish():
            action = agent.getAction(simu)
            simu.takeAction(action)
        z+= 1.0 if view == simu.getWinner() else -1.0
    return z/cnt


def randomRolloutFn(simulator):
    view = simulator.getCurrentPlayer()
    z = 0
    cnt = 30
    for i in range(cnt):
        simu = copy.deepcopy(simulator)
        while not simu.isFinish():
            action = random.choice(simu.getAvailableActions())
            simu.takeAction(action)
        z+= 1.0 if view == simu.getWinner() else -1.0
    return z/cnt