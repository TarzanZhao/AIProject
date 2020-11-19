import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import Board
import Game
import Agent
import numpy as np


class GUI(QWidget):
    def __init__(self, agent1,agent2,boardSize=15, numberForWin=5):
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
        #self.RestartVisualRun = QPushButton("RestartVisual",self)
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
        for action, c in self.Board.actions:
            TL = self.posToTopleft(action)
            qp.setBrush(Qt.black if c == 1 else Qt.white)
            rec = QRect(TL[0],TL[1],int(2*self.gap+0.5),int(2*self.gap+0.5))
            qp.drawEllipse(rec)
        qp.end()

    def mousePressEvent(self, e):
        if self.win:
            return
        pos = self.CoordinateTopos(e.x(), e.y())
        #QMessageBox.information(self, '框名', "(%d,%d)" %(pos[0],pos[1]))
        if (pos[0] < 0) or (pos[0] >= self.S) or (pos[1] < 0) or (pos[1] >= self.S) or (self.Board.encodeAction(pos) not in self.Board.available):
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
        self.Board.tackAction(pos)
        self.C = 1 - self.C
        self.update()
        if self.Board.end():
            self.win=1
            QMessageBox.information(self, 'result', "agent 1 win!" if self.Board.winner==1 else "agent 2 win!")

    def run(self):
        if self.win:
            return 0
        action = self.agent2.getAction(self.Board)
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


app = QApplication(sys.argv)
#gui = GUI(agent1=A1,agent2=A2)
#gui.visualizeRun(A1,A2)
sys.exit(app.exec_())
