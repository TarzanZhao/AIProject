import torch
import numpy
import json
import os

def saveData(dataList, file):
    """
    :param dataList: [(4*8*8 3-d tensor, 8*8+1 1-d tensor, 1 1-d tensor),...]
    :param file: file to store the data.
    :return:
    """
    dataForSave = []
    for data in dataList:
        dataForSave.append((data[0].numpy().tolist(), data[1].numpy().tolist(), data[2].numpy().tolist()))
    with open(file, "w") as F:
        F.write(json.dumps(dataForSave))

def retrieveData(file):
    """
    :param file:
    :return: dataList: [(4*8*8 3-d tensor, 8*8+1 1-d tensor, 1 1-d tensor),...]
    """
    dataList = []
    F = open(file, "r")
    dataString = F.readline().strip("\n")
    F.close()
    rawData = json.loads(dataString)
    for data in rawData:
        dataList.append(tuple([torch.tensor(data[0]), torch.tensor(data[1]), torch.tensor(data[2],dtype=torch.float)]))
    return dataList

def getLatestNetworkID():
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