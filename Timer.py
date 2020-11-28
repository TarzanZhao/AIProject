import time
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

class Timer:
    def __init__(self):
        self.timeTable = {}

    def clear(self):
        self.timeTable = {}

    def startTime(self, name):
        if not name in self.timeTable:
            self.timeTable[name] = 0
        self.timeTable[name] -= time.time()
        return name

    def endTime(self, name):
        self.timeTable[name] += time.time()

    def getTime(self, name):
        if not name in self.timeTable:
            return -1
        return self.timeTable[name]

    def showTime(self, name):
        if not name in self.timeTable:
            print("Wrong name!")
            return
        elapsed_time = self.timeTable[name]
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        print("Time for \"" + name + "\" is: %d m %d s" % (elapsed_mins, elapsed_secs))

    def showTimeInfo(self, title):
        labels = list(self.timeTable.keys())
        data = [self.timeTable[t] for t in labels]

        # initial figure
        fig, axes = plt.subplots(figsize=(10,5) ,ncols=2)
        ax1, ax2 = axes.ravel()

        def showPCT(pct):
            return ('%.2f' % pct) if pct > 5 else ''
        # set color
        colors = cm.rainbow(np.arange(len(data)) / len(data))  # colormaps: Paired, autumn, rainbow, gray,spring,Darks
        patches, texts, autotexts = ax1.pie(data, labels=None, autopct=showPCT,
                                           shadow=False, colors=colors,)
        ax1.axis('equal')
        ax1.set_title(title)
        ax2.axis('off')
        ax2.legend(patches, labels, loc='center left')
        plt.tight_layout()
        plt.show()
        plt.close()

        plt.figure()
        plt.bar(labels,data)
        plt.show()
        plt.close()


timer = Timer()



