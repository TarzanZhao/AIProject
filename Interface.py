import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import argparse
import PolicyValueFn
import Board
import Game
import Agent
import numpy as np


class GUI(QWidget):
    def __init__(self, agent1,agent2,boardSize=15, numberForWin=5):
        super().__init__()
        self.C = 0  # 0: black, 1: white
        self.S = boardSize
        self.numberForWin = numberForWin
        self.W = 700
        self.B = 50
        self.d = self.W / (self.S - 1)
        self.gap = 0.4 * self.d
        self.Board = Board.Board(boardSize,numberForWin)
        self.Board.init()
        self.agent1 =agent1
        self.agent2 =agent2
        self.win = 0
        self.Restart = QPushButton("Restart",self)
        self.Restart.clicked.connect(self.OnRestart)
        self.RestartVisualRun = QPushButton("RestartVisual",self)
        self.RestartVisualRun.clicked.connect(self.OnRestartVisual)
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
            rec = QRect(TL[0],TL[1],int(2*self.gap+0.5),int(2*self.gap+0.5))
            qp.drawEllipse(rec)
            c = 1-c
        qp.end()

    def mousePressEvent(self, e):
        if self.win:
            return
        pos = self.CoordinateTopos(e.x(), e.y())
        #QMessageBox.information(self, '框名', "(%d,%d)" %(pos[0],pos[1]))
        if (pos[0] < 0) or (pos[0] >= self.S) or (pos[1] < 0) or (pos[1] >= self.S) or (pos not in self.Board.getAvailableActions()):
            return
        else:
            self.down(pos)
            self.run()

    def initUI(self):
        self.resize(1200, 800)
        self.move(30, 80)
        self.setWindowTitle('Chess Board')
        self.show()
        self.Restart.setGeometry(QRect(900,200,200,100))
        self.RestartVisualRun.setGeometry(QRect(900,600,200,100))

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
        if self.Board.isFinish():
            self.agent1.isFinish(1)
            self.win=1
            QMessageBox.information(self, 'result', "agent 0 win!" if self.Board.getWinner()==0 else "agent 1 win!")

    def run(self):
        if self.win:
            return 0
        action = self.agent1.getAction(self.Board)
        self.down(action)


    def OnRestart(self):
        self.Board.init()
        self.win = 0
        self.update()

    def OnRestartVisual(self):
        self.Board.init()
        self.agent1.init()
        self.agent2.init()
        self.visualizeRun(self.agent1,self.agent2)



parser = argparse.ArgumentParser()
parser.add_argument('--train-batch-size', type=int, default=15000)
parser.add_argument('--channels',type=int,default=4)
parser.add_argument('--size',type=int,default=8)
parser.add_argument('--numOfIterations',type=int,default=1600)
parser.add_argument('--numberForWin',type=int,default=4)

parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--show-size', type=int, default=15000)
parser.add_argument('--std', type=float, default=0.01)
parser.add_argument('--show', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

model = PolicyValueFn.PolicyValueFn(args)
agent1 = Agent.SelfplayAgent(args.numOfIterations,model,"test.txt")
agent2 = Agent.SelfplayAgent(args.numOfIterations,model,"test.txt")
b = Board.Board(args.size,args.numberForWin)

app = QApplication(sys.argv)
gui = GUI(agent1=agent1,agent2=agent2,boardSize = args.size,numberForWin=args.numberForWin)
#gui.visualizeRun(agent1,agent2)
sys.exit(app.exec_())
