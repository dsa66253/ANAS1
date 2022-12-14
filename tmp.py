from os import listdir
from os.path import isfile, join
import os
import pstats
from re import T
import sys
import torch
from  models.mymodel import Model
from  models.retrainModel import NewNasModel
import json
from utility.HistDrawer import HistDrawer
from models.arch import simpleArch
from models.Layer import Layer
'''
pick up 140 image from training set to test set
'''
def getAllFileName(dirPath):
    return [f for f in listdir(dirPath) if isfile(join(dirPath, f))]
def deleteFile(filePath):
    if os.path.exists(filePath):
        os.remove(filePath)
def moveFile(sourceDir, desDir, fileName):
    sourcePath = os.path.join(sourceDir, fileName)
    desPath = os.path.join(desDir, fileName)
    os.rename(sourcePath, desPath)
def setStdoutToFile(filePath):
    print("std output to ", filePath)
    f = open(filePath, 'w')
    sys.stdout = f
    return f

if __name__=="__main__":
    layer = Layer(1, 1, cellArchPerLayer=[[1, 1, 1, 1, 1]])
    # print(layer.getAlphas())
    # print(id(layer.getBeta()[0]))
    # for k, v in layer.named_parameters():
    #     print(k)
    for k, v in layer.named_modules():
        # print(k)
        print(v)
        break
    print("==========================")
    for k, v in layer.named_children():
        print(k)
        print(v)
        break
    exit()
    histDrawer = HistDrawer("./")
    # op = OPS["conv_5x5"](96, 128, 1, 1, 1)
    net = Model()
    
    histDrawer.drawNetConvWeight(net)
    # currentAlpha = torch.FloatTensor([0.1, 0.2, 0.5, 0.4, -0.3])
    # currentAlpha = torch.abs(currentAlpha)
    # (_, allMinIndex) = torch.topk( currentAlpha, 5, largest=False )
    # print(_, allMinIndex, torch.abs(currentAlpha))
    
    exit()
    print(torch.cuda.memory_allocated(device="cuda"))
    l = []
    for i in range(5):
        tmp = torch.rand(100, 100).to("cuda")
        # print(tmp.grad)
        l.append(tmp)
        print(torch.cuda.memory_allocated(device="cuda"))
        tmp.requires_grad_(True)
        print(torch.cuda.memory_allocated(device="cuda"))
        print(tmp.grad)
    # print(l)
    exit()
    setStdoutToFile("./tmp.txt")
    f = open("./decode/0th_decode.json")
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
    # print(data)
    for key in data:
        print(key, data[key])
    model = NewNasModel(data)
    print(model)
    x = torch.rand(5, 3, 128, 128)
    model(x)
    exit()

    torch.manual_seed(10)
    input = torch.rand(3, 3, 128, 128)
    net = Model()
    output = net(input)
    output = output.sum()
    output.backward()
    # output = net(input)
    # print(.shape)
    exit()



    torch.manual_seed(10)
    input = torch.rand(3, 3, 64, 64)
    net = Layer(1, 0, 3, 96, 1, 1, [0, 1],  "testLayer")
    output = net(input)
    output = output.sum()
    output.backward()
    output = net(input)
    # print(.shape)
    exit()





    torch.manual_seed(10)
    input = torch.rand(3, 3, 64, 64)
    innercell = InnerCell(3, 96, 1, None, "testLayer")
    output = innercell(input)
    output = output.sum()
    output.backward()
    output = innercell(input)
    # print(.shape)
    exit()



    sourceDir = "../dataset/train"
    desDir = "../dataset/test"

    sourceClassDirList = [x[0] for x in os.walk("../dataset/train")][1:]
    desClassDirList = [x[0] for x in os.walk("../dataset/test")][1:]
    for i in range(len(sourceClassDirList)):
        fileList = getAllFileName(sourceClassDirList[i])
        for j in range(140):
            # pass
            moveFile(sourceClassDirList[i], desClassDirList[i], fileList[j])
    