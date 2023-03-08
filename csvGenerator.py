import csv, json, os
from data.config import folder
import numpy as np
class csvGenerator():
    def __init__(self, expName, path, keyList) -> None:
        self.path = path
        self.expName = expName
        self.keyList = keyList
        self.expectHeader = ["expName", "kth", "layer0_1", "layer0_2", "testAcc", "valAcc"]
    def writeHeader(self):
        with open(self.path, 'a', encoding='UTF8') as f:
            writer = csv.DictWriter(f, fieldnames=self.keyList)
            writer.writeheader()
    def appendRow(self, dicRows:list):
        with open(self.path, 'a', encoding='UTF8') as f:
            writer = csv.DictWriter(f, fieldnames=self.keyList)
            # write the data
            writer.writerow(dicRows)
def getAccByMaxVal(self, i, j, k, baseDir):
    valAcc = np.load( "./log/{}/accLoss/retrain_val_acc_{}.npy".format(baseDir, str(k)) )
    testAcc = np.load("./log/{}/accLoss/retrain_test_acc_{}.npy".format(baseDir, str(k)) )
    valIndex = np.argmax(valAcc)
    return round(testAcc[valIndex], 2)
if __name__=="__main__":
    expNameList = [""]
    for expName in expNameList:
        
        kth = 0
        filePath = os.path.join(folder["decode"], "{}th_decode.json".format(kth))
        f = open(filePath)
        archDict = json.load(f)
        keys = list(archDict.keys())
        csvG = csvGenerator("test", "./countries.csv", keys)
        csvG.appendRow(archDict)
    
