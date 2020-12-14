import Agent
import Board
import Game
import matplotlib.pyplot as plt
from DataStorage import dataProcessor
import argument
import logger
import os
from RolloutFn import randomRolloutFn,minMaxRolloutFn

"""
main experiment
(1) For given model, do comparison for four types of rollout: 
None("sign" of feature), randomRolloutFn, minMaxRolloutFn, return_v_value 
-> function: evaluationForNetworkWithFourRollout()
(2) to be done
"""

class Experiment():
    def __init__(self):
        self.args = argument.get_args()
        self.logger = logger.get_logger()
        self.depth = 0
        self.numOfEvaluations = self.args.numOfEvaluations
        self.evaluationData= []

    def evaluationWithBaseLine(self,agent, agentFirstOnly = False, numOfEvaluations = None):
        """
        :return: Winning rate for agent to baseline
        """
        if numOfEvaluations is not None:
            self.numOfEvaluations = numOfEvaluations
        #agent2 = Agent.SearchAgent(depth=self.depth,epsilon=0)
        agent2 = Agent.RandomAgent()
        return self.evaluation(agent,agent2,agentFirstOnly)

    def evaluationWithDifferentMinMaxSearchAgent(self,agent,agentFirstOnly = False, numOfEvaluations = None):
        """
        :return: return winning rate for {depth: winrate} (0~4)
        """
        if numOfEvaluations is not None:
            self.numOfEvaluations = numOfEvaluations
        dic = {}
        for i in range(0,5):
            agent2 = Agent.SearchAgent(depth=i, epsilon=0)
            dic[i] = self.evaluation(agent,agent2,agentFirstOnly=agentFirstOnly)

    def evaluationForNetworkWithDifferentSearchDepth(self,model, start= 5,end = 405, step = 50):
        """
        bare network <-> MCTS network: the picture is saved
        :return: two lists: numOfIterations and winning rate (in the view of MCTS)
        """
        numOfIterations = list(range(start,end,step))
        winRate = []
        agent1 = Agent.NetWorkAgent(model)
        for iter in numOfIterations:
            agent2 = Agent.IntelligentAgent(iter,model, balance=0)
            winRate.append(self.evaluation(agent1,agent2))

        fig = self.simplePlot(numOfIterations,winRate,
                        title='Effect of MCTS',xlabel='num of iteration',ylabel='wining rate of MCTS')
        self.saveFig(fig,name="Effect of MCTS")
        return numOfIterations,winRate

    def evaluationWithBaselineInDifferentNumOfIter(self, agent, start = 100, end = 500, step = 100, numOfEvaluations=30):
        """
        :return: two lists: numOfIterations, winRate
        """
        numOfIterations = list(range(start, end, step))
        winRate = []
        for iter in numOfIterations:
            agent.setNumOfIterations(iter)
            winRate.append(self.evaluationWithBaseLine(agent,numOfEvaluations=numOfEvaluations))
        return numOfIterations,winRate

    def evaluationForNetworkWithFourRollout(self, model, random_cnt = 7, minmax_cnt =1,start= 100,end = 500, step = 100,numOfEvaluations=30):
        """
        :return: {type: {num of iterations: winRate}} fig saves
        """
        type = ['sign_feature','ran_roll','mm_roll','net_v']
        ranRoll = randomRolloutFn(random_cnt)
        mmRoll = minMaxRolloutFn(minmax_cnt)
        agents = [Agent.IntelligentAgent(0,model,None,balance=1),
                  Agent.IntelligentAgent(0,model,rolloutFn=ranRoll,balance=1),
                  Agent.IntelligentAgent(0,model,rolloutFn=mmRoll,balance=1),
                  Agent.IntelligentAgent(0,model,balance=0)]
        color = ['blue','pink','red','yellow']
        data = {}
        fig = None
        for i,t in enumerate(type):
            data[t] = self.evaluationWithBaselineInDifferentNumOfIter(agents[i],start=start,end=end,step=step,numOfEvaluations=numOfEvaluations,)
            fig = self.simpleBar(data[t][0],data[t][1],width=0.24,title='Comparison for 4 Types of Rollouts'
                                 ,xlabel='num of iterations',ylabel='winning rate to baseline',
                                 color=color[i],label=t,fig = fig)
        fig.legend(loc = 2)
        self.saveFig(fig,name='Comparison for 4 Types of Rollouts')
        return data

    def evaluation(self, agent1, agent2, agentFirstOnly=False):
        board = Board.Board(self.args.size, self.args.numberForWin)
        game = Game.Game(agent1, agent2, board)
        Score = {agent1: 0, agent2: 0}
        self.logger.info("First Player Case:")
        for i in range(1, 1 + self.numOfEvaluations):
            winner = game.run()
            # self.evaluationData.append(game.getActionSequence())
            Score[winner] += 1
            if winner == agent1:
                self.logger.info("The %dth game: Win!" % i)
            else:
                self.logger.info("The %dth game: Lose!" % i)
        if not agentFirstOnly:
            game = Game.Game(agent2, agent1, board)
            self.logger.info("Second Player Case:")
            for i in range(1, 1 + self.numOfEvaluations):
                winner = game.run()
                # self.evaluationData.append(game.getActionSequence())
                Score[winner] += 1
                if winner == agent1:
                    self.logger.info("The %dth game: Win!" % (i + self.numOfEvaluations))
                else:
                    self.logger.info("The %dth game: Lose!" % (i + self.numOfEvaluations))
        return Score[agent1] / (2 * self.numOfEvaluations - self.numOfEvaluations * agentFirstOnly)

    def simplePlot(self, X, Y, title, xlabel="x", ylabel='y', color='blue', linestyle='-.', label = None,fig=None, ):
        if fig is None:
            fig = plt.figure(figsize=(15,8))
        plt.plot(X, Y, color=color, linestyle=linestyle,label = label)
        plt.title(title,fontsize=25)
        plt.xlabel(xlabel,fontsize=17)
        plt.ylabel(ylabel,fontsize=17)
        return fig

    def simpleBar(self, X, Y, title, width = 1.0,xlabel = 'x', ylabel = 'y',color = 'green',label = None, fig = None):
        if fig is None:
            fig = plt.figure(figsize=(15,8))
        plt.bar(X,Y,width=width, facecolor = color,alpha =0.6, label = label)
        plt.title(title,fontsize=25)
        plt.xlabel(xlabel,fontsize=17)
        plt.ylabel(ylabel,fontsize=17)
        return fig

    def saveFig(self,fig, name):
        fig.savefig(os.path.join(self.args.figure_folder, name + '.png'))
        plt.close(fig)
