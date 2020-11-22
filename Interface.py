import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import argparse
import PolicyValueFn
import Board
import Game
import Agent
from DataStorage import getLatestNetworkID
import numpy as np
import torch


class GUI(QWidget):
    def __init__(self, agent, boardSize=15, numberForWin=5):
        super().__init__()
        self.C = 0  # 0: black, 1: white
        self.S = boardSize
        self.numberForWin = numberForWin
        self.W = 700
        self.B = 50
        self.d = self.W / (self.S - 1)
        self.gap = 0.4 * self.d
        self.Board = Board.Board(boardSize, numberForWin)
        self.Board.init()
        self.win = 0
        self.agent = agent
        self.Restart = QPushButton("Restart", self)
        self.Restart.clicked.connect(self.OnRestart)
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
        for action in self.Board.actions:
            TL = self.posToTopleft(action)
            qp.setBrush(Qt.black if c == 1 else Qt.white)
            rec = QRect(TL[0], TL[1], int(2 * self.gap + 0.5), int(2 * self.gap + 0.5))
            qp.drawEllipse(rec)
            c = 1 - c
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
            self.run()

    def initUI(self):
        self.resize(1200, 800)
        self.move(30, 80)
        self.setWindowTitle('Chess Board')
        self.show()
        self.Restart.setGeometry(QRect(900, 200, 200, 100))
        #self.RestartVisualRun.setGeometry(QRect(900, 600, 200, 100))

    def drawBackground(self, qp):
        pen = QPen(Qt.black, 2, Qt.SolidLine)
        qp.setPen(pen)
        s = np.linspace(0, self.W, self.S)
        for i in range(self.S):
            qp.drawLine(s[i] + self.B, self.B, s[i] + self.B, self.W + self.B)
            qp.drawLine(self.B, s[i] + self.B, self.W + self.B, s[i] + self.B)

    def down(self, pos):
        self.Board.takeAction(pos)
        self.C = 1 - self.C
        self.update()
        self.repaint()
        if self.Board.isFinish():
            self.win = 1
            QMessageBox.information(self, 'result', "agent 0 win!" if self.Board.getWinner() == 0 else "agent 1 win!")

    def run(self):
        if self.win:
            return 0
        action = self.agent.getAction(self.Board)
        self.down(action)

    def OnRestart(self):
        self.Board.init()
        self.win = 0
        self.update()

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


def Play(args):
    model = PolicyValueFn.PolicyValueFn(args)
    currentModel = getLatestNetworkID()
    model.load_state_dict(torch.load(f'network/network-{currentModel}.pt',map_location=torch.device('cpu')))
    agent = Agent.IntelligentAgent(args.numOfIterations,model)
    app = QApplication(sys.argv)
    gui = GUI(agent, boardSize=args.size, numberForWin=args.numberForWin)
    sys.exit(app.exec_())