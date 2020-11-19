import torch
import numpy
import json

def saveData(dataList, file):
    """
    :param dataList: [(4*8*8 3-d tensor, 8*8+1 1-d tensor, 1 1-d tensor),...]
    :param file: file to store the data.
    :return:
    """
    dataForSave = []
    for data in dataList:
        print(len(data))
        print(data[0].size(),data[1].size())
        print(data[2].size())

        dataForSave.append( (data[0].numpy().tolist(), data[1].numpy().tolist(), data[2].numpy().tolist()) )
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
        dataList.append( (torch.Tensor(data[0]), torch.Tensor(data[1]), torch.Tensor(data[2])) )
    return dataList