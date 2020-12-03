import torch
from PolicyValueFn import PolicyValueFn
import json
import os

class DataProcessor:
    def __init__(self):
        self.simulator = None

    def initSimulator(self,simulator):
        self.simulator = simulator

    def saveData(self, dataList, file):
        """
        :param dataList: [(action, 8*8 1-d tensor, 1 1-d tensor),"end",...] "end": end of a game
        :param file: file to store the data.
        :return:
        """
        print("Try saving data into "+file+".")
        dataForSave = []
        for data in dataList:
            if data == 'end':
                dataForSave.append('end')
            else:
                dataForSave.append((data[0], data[1], data[2]))
        with open(file, "w") as F:
            F.write(json.dumps(dataForSave))
        print("Successfully save data into "+file+".")

    def retrieveData(self, file):
        """
        :param file:
        :return: dataList: [(4*8*8 3-d tensor, 8*8+1 1-d tensor, 1 1-d tensor),...]
        """
        self.simulator.init()
        dataList = []
        F = open(file, "r")
        dataString = F.readline().strip("\n")
        F.close()
        rawData = json.loads(dataString)
        nPlay = 0
        for data in rawData:
            # (action, policy, value)
            if data == 'end':
                nPlay += 1
                self.simulator.init()
            else:
                dataList.append(tuple([torch.tensor(self.simulator.getCurrentState(),dtype=torch.float),
                                       torch.tensor(data[1]), torch.tensor(data[2], dtype=torch.float)]))
                self.simulator.takeAction(tuple(data[0]))
        print(f"load {nPlay} plays' data")
        return dataList

    def getLatestNetworkID(self):
        path = './network'
        files = os.listdir(path)
        currentModel = -1
        for file in files:
            if not os.path.isdir(file):
                filestr = file.split("/")[-1]
                print(filestr)
                if filestr.startswith("network-"):
                    idstr = filestr[8:-3]
                    print(idstr)
                    currentModel = max(currentModel, int(idstr))
        return currentModel

    def loadNetwork(self, args, currentModelID=None):
        model = PolicyValueFn(args)
        if currentModelID is None:
            currentModelID = self.getLatestNetworkID()
        model.load_state_dict(torch.load(f'network/network-{currentModelID}.pt', map_location=torch.device(args.device)))
        model.to(args.device)
        return model

    def getLastestSelfplay(self):
        path = './selfplay'
        files = os.listdir(path)
        latestSelfplay = -1
        for file in files:
            if not os.path.isdir(file):
                filestr = file.split("/")[-1]
                print(filestr)
                if filestr.startswith("selfplay-"):
                    idstr = filestr[9:-4]
                    print(idstr)
                    latestSelfplay = max(latestSelfplay, int(idstr))
        if latestSelfplay == -1:
            return []
        else:
            file = f"selfplay/selfplay-{latestSelfplay}.txt"
            dataList = []
            dataList.append([])
            num = 0
            F = open(file, "r")
            dataString = F.readline().strip("\n")
            F.close()
            rawData = json.loads(dataString)
            for data in rawData:
                if data == 'end':
                    num += 1
                    dataList.append([])
                else:
                    dataList[num].append(tuple(data[0]))
            dataList.pop()
        return dataList

dataProcessor = DataProcessor()