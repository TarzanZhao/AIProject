class Agent:
    def getAction(self, simulator):
        """
        :param simulator: a game state simulator (can do query)
        :return: an available action (x,y)
        """
        pass

    def finish(self, isWin):
        """
        :param isWin: is the Agent win
        """
        pass

    def init(self):
        pass



class SelfplayAgent(Agent):
    def __init__(self):
        pass

    def __str__(self):
        return "SelfplayAgent Instance"


class RandomAgent(Agent):
    def __init__(self):
        pass

    def __str__(self):
        return "RandomAgent Instance"

class HumanAnent(Agent):
    def __init__(self):
        pass

    def __str__(self):
        return "HumanAgent Instance"

class IntelligentAgent(Agent):
    def __init__(self):
        pass

    def __str__(self):
        return "IntelligentAgent Instance"