import torch
from PolicyValueFn import PolicyValueFn
import json
import os
import argument
import logger

class DataProcessor:
    def __init__(self):
        self.simulator = None

    def initSimulator(self,simulator):
        self.simulator = simulator
        self.logger = logger.get_logger()
        self.args = argument.get_args()

    def saveData(self, dataList, file):
        """
        :param dataList: [(action, 8*8 1-d tensor, 1 1-d tensor),"end",...] "end": end of a game
        :param file: file to store the data.
        :return:
        """
        self.logger.info("Try saving data into "+file+".")
        dataForSave = []
        for data in dataList:
            if data == 'end':
                dataForSave.append('end')
            else:
                dataForSave.append((data[0], data[1], data[2]))
        with open(file, "w") as F:
            F.write(json.dumps(dataForSave))
        self.logger.info("Successfully save data into "+file+".")

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
        self.logger.info(f"load {nPlay} plays' data")
        return dataList

    def getLatestNetworkID(self):
        path = self.args.model_folder
        files = os.listdir(path)
        currentModel = -1
        for file in files:
            if not os.path.isdir(file):
                filestr = file.split("/")[-1]
                self.logger.info(filestr)
                if filestr.startswith("network-"):
                    idstr = filestr[8:-3]
#                    print(idstr)
                    currentModel = max(currentModel, int(idstr))
        return currentModel

    def loadNetwork(self, args, currentModelID=None):
        model = PolicyValueFn(args)
        if currentModelID is None:
            currentModelID = self.getLatestNetworkID()
        model.load_state_dict(torch.load(os.path.join(self.args.model_folder, f'network-{currentModelID}.pt'), map_location=torch.device(args.device)))
        model.to(args.device)
        return model

    def getLastestSelfplay(self):
        path = self.args.data_folder
        files = os.listdir(path)
        latestSelfplay = -1
        for file in files:
            if not os.path.isdir(file):
                filestr = file.split("/")[-1]
                print(filestr)
                if filestr.startswith("selfplay-"):
                    idstr = filestr[9:-4]
#                    print(idstr)
                    latestSelfplay = max(latestSelfplay, int(idstr))
        if latestSelfplay == -1:
            return []
        else:
            file = os.path.join(path, f"selfplay-{latestSelfplay}.txt")
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

    def getLastestSearchPlay(self):
        path = os.path.join(self.args.data_root, self.args.load_data_folder)
        files = os.listdir(path)
        latestSearchPlay = -1
        for file in files:
            if not os.path.isdir(file):
                filestr = file.split("/")[-1]
                print(filestr)
                if filestr.startswith("searchPlay-"):
                    idstr = filestr[11:]
#                    print(idstr)
                    latestSearchPlay = max(latestSearchPlay, int(idstr))
        if latestSearchPlay == -1:
            return []
        else:
            file = os.path.join(path,f"searchPlay-{latestSearchPlay}")
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