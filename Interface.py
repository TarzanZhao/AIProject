import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import PolicyValueFn
import Board
import Agent
import numpy as np
import torch
from Timer import timer
from DataStorage import dataProcessor


class GUI(QWidget):
    def __init__(self, agent, boardSize=15, numberForWin=5, agentFirst = 0):
        super(GUI, self).__init__()
        self.S = boardSize
        self.numberForWin = numberForWin
        self.agentFirst = agentFirst
        self.W = 700
        self.B = 50
        self.d = self.W / (self.S - 1)
        self.gap = 0.4 * self.d
        self.Board = Board.Board(boardSize, numberForWin)
        self.Board.init()
        self.win = 0
        self.agent = agent
        #if str(self.agent)=="SearchAgent Instance":
        #    self.Board.setFeatureUpdate(True)
        #else:
        #    self.Board.setFeatureUpdate(False)
        self.Restart = QPushButton("Restart", self)
        self.ShowValue = QPushButton("ShowValue",self)
        self.RandomSelfplay = QPushButton("RandomVisualize",self)
        self.RandomSearchPlay = QPushButton("SearchPlayVisualize",self)
        self.Restart.clicked.connect(self.OnRestart)
        self.ShowValue.clicked.connect(self.OnShowValue)
        self.RandomSelfplay.clicked.connect(self.OnRandomSelfplay)
        self.RandomSearchPlay.clicked.connect(self.OnRandomSearchPlay)
        self.value = 1
        self.isShowValue = 1
        self.isShowSelfplay = 0
        self.isShowSearchPlay = 0
        self.policy = {}
        self.searchPlayData = None
        self.selfplayData = None
        #self.RestartVisualRun = QPushButton("RestartVisual", self)
        #self.RestartVisualRun.clicked.connect(self.OnRestartVisual)
        self.initUI()

    def posToTopleft(self, pos):
        return self.B + pos[0] * self.d - self.gap, self.B + pos[1] * self.d - self.gap

    def CoordinateTopos(self, x, y):
        return int((x - self.B) / self.d + 0.5), int((y - self.B) / self.d + 0.5)

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.drawBackground(qp)
        c = 0
        if self.isShowValue:
            for action, pro in self.policy.items():
                TL = self.posToTopleft(action)
                blue = QColor(183, 234, 255)
                qp.setPen(blue)
                qp.setBrush(blue)
                rec = QRect(TL[0], TL[1], int(2 * self.gap + 0.5), int(2 * self.gap + 0.5))
                qp.setPen(Qt.black)
                qp.drawEllipse(rec)
                qp.setFont(QFont('SimHei', int(80/self.Board.getSize())))
                qp.drawText(int(TL[0]+self.gap/2+0.5),int(TL[1]+self.gap+0.5),"%.2f"%pro)

        num = 1
        for action in self.Board.actions:
            TL = self.posToTopleft(action)
            qp.setPen(Qt.black)
            qp.setBrush(Qt.black if c == 1 else Qt.white)
            rec = QRect(TL[0], TL[1], int(2 * self.gap + 0.5), int(2 * self.gap + 0.5))
            qp.drawEllipse(rec)
            qp.setPen(Qt.black if c == 0 else Qt.white)
            qp.setFont(QFont('Arial',int(160/self.Board.getSize())))
            if num>=100:
                qp.drawText(int(TL[0]+self.gap/4+0.5),int(TL[1]+self.gap*1.2+0.5),"%d" %num)
            elif num>=10 :
                qp.drawText(int(TL[0] + self.gap/2+0.5), int(TL[1] + self.gap*1.2 + 0.5), "%d" %num)
            else :
                qp.drawText(int(TL[0] + self.gap/1.4 + 0.5), int(TL[1] + self.gap * 1.2 + 0.5), "%d" % num)
            num +=1
            c = 1 - c
        if self.isShowValue and not self.isShowSelfplay and not self.isShowSearchPlay and len(self.Board.actions)>0:
            last = self.Board.actions[-1]
            TL = self.posToTopleft(last)
            gold = QColor(255,255,0)
            qp.setPen(QPen(gold,5))
            qp.setBrush(Qt.black if c==0 else Qt.white)
            rec = QRect(TL[0], TL[1], int(2 * self.gap + 0.5), int(2 * self.gap + 0.5))
            qp.drawEllipse(rec)
            if c==self.agentFirst:
                qp.setPen(Qt.black)
                qp.setFont(QFont('SimHei', int(80/self.Board.getSize())))
                qp.drawText(int(TL[0] +self.gap/2+ 0.5), int(TL[1] + self.gap + 0.5), "%.2f" % self.policy[last])
        qp.end()

    def mousePressEvent(self, e):
        if self.win:
            return
        pos = self.CoordinateTopos(e.x(), e.y())
        # QMessageBox.information(self, '框名', "(%d,%d)" %(pos[0],pos[1]))
        if (pos[0] < 0) or (pos[0] >= self.S) or (pos[1] < 0) or (pos[1] >= self.S) or (
                pos not in self.Board.getAvailableActions()):
            return
        else:
            self.down(pos)
            self.update()
            self.repaint()
            self.run()

    def initUI(self):
        self.resize(1200, 800)
        self.move(30, 80)
        self.setWindowTitle('Chess Board')
        self.show()
        self.Restart.setGeometry(QRect(900, 50, 200, 100))
        self.ShowValue.setGeometry(QRect(900,250,200,100))
        self.RandomSelfplay.setGeometry(QRect(900,450,200,100))
        self.RandomSearchPlay.setGeometry(QRect(900,650,200,100))
        #self.RestartVisualRun.setGeometry(QRect(900, 600, 200, 100))
        if self.agentFirst:
            self.run()

    def drawBackground(self, qp):
        pen = QPen(Qt.black, 2, Qt.SolidLine)
        qp.setPen(pen)
        s = np.linspace(0, self.W, self.S)
        for i in range(self.S):
            qp.drawLine(s[i] + self.B, self.B, s[i] + self.B, self.W + self.B)
            qp.drawLine(self.B, s[i] + self.B, self.W + self.B, s[i] + self.B)

    def down(self, pos):
        self.Board.takeAction(pos)
        self.update()
        if self.Board.isFinish():
            self.win = 1
            QMessageBox.information(self, 'result', "agent 0 win!" if self.Board.getWinner() == 0 else "agent 1 win!")

    def run(self):
        if self.win:
            return 0
        #print(f"In the view of {self.Board.getCurrentPlayer()}: {self.Board.approxScore()}")
        action = self.agent.getAction(self.Board)
        self.policy = self.agent.getActionProPair()
        self.down(action)
        #print(f"In the view of {self.Board.getCurrentPlayer()}: {self.Board.approxScore()}")

    def clear(self):
        self.Board.init()
        self.win = 0
        self.policy = {}
        self.update()

    def showPlay(self, play):
        for action in play:
            self.Board.takeAction(action)


    def OnRestart(self):
        self.isshowRandom = 0
        self.clear()
        if self.agentFirst:
            self.run()

    def OnShowValue(self):
        self.isShowValue = 1 - self.isShowValue
        self.update()
        self.repaint()

    def OnRandomSelfplay(self):
        self.isShowSelfplay = 1
        if self.selfplayData == None:
            self.selfplayData = dataProcessor.getLastestSelfplay()
        index = np.random.randint(0,len(self.selfplayData))
        play = self.selfplayData[index]
        self.clear()
        self.showPlay(play)
        print(play)
        self.update()
        self.win = 1

    def OnRandomSearchPlay(self):
        self.isShowSearchPlay = 1
        if self.searchPlayData is None:
            self.searchPlayData = dataProcessor.getLastestSearchPlay()
        index = np.random.randint(0,len(self.searchPlayData))
        play = self.searchPlayData[index]
        print(play)
        self.clear()
        self.showPlay(play)
        self.update()
        self.win = 1
    #def OnRestartVisual(self):
    #    self.Board.init()
    #    self.agent.init()
    #    self.agent2.init()
    #    self.visualizeRun(self.agent1, self.agent2)
    #def visualizeRun(self, agent0, agent1):
    #    self.Board.init()
    #    agentMap = {0: agent0, 1: agent1}
    #    while not self.Board.isFinish():
    #        agent = agentMap[self.Board.getCurrentPlayer()]
    #        action = agent.getAction(self.Board)
    #        if action in self.Board.getAvailableActions():
    #            self.down(action)
    #    winner = self.Board.getWinner()
    #    agent0.finish(winner == 0)
    #    agent1.finish(winner == 1)
    #    return agentMap[winner]

def IntelligenceAgent(args, modelID=None):
    model = dataProcessor.loadNetwork(args, modelID)
    agent = Agent.IntelligentAgent(args.numOfIterations, model)
    return agent

def NetworkAgent(args):
    model = dataProcessor.loadNetwork(args)
    agent = Agent.NetWorkAgent(model)
    return agent

def Play(args, agent):
    app = QApplication(sys.argv)
    gui = GUI(agent, boardSize=args.size, numberForWin=args.numberForWin, agentFirst=args.agentFirst)
    sys.exit(app.exec_())


