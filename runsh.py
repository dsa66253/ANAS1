import subprocess
import json, os, sys, copy
from os.path import isfile, join
from data.config import folder, cfg_nasmodel as cfg, seed
def makeAllDir():
    for folderName in folder:
        print("making folder ", folder[folderName])
        makeDir(folder[folderName])
def makeDir(folderPath):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
def setStdoutToFile(filePath):
    f = open(filePath, 'w')
    sys.stdout = f
    return f
def setStdoutToDefault(f):
    f.close()
    sys.stdout = sys.__stdout__
def doExpBasedExperiments():
    conti = True
    while conti:
        filePath = os.path.join("./experiments.json")
        f = open(filePath)
        exp = json.load(f)
        finishCount = 0
        for expName in exp:
            if exp[expName]==0:
                # exp[expName]=1
                # f = setStdoutToFile("./experiments.json")
                # print(json.dumps(exp, indent=4)) #* make ndarray to list
                # setStdoutToDefault(f)
                subprocess.call('./train.sh')
                break
            finishCount = finishCount + 1
            if finishCount==len(exp):
                exit()
        print("finish trina.sh")
def brutNas():
    for kth in range(cfg["numOfKth"]):

        f = setStdoutToFile("./curExperiment.json")
        curExpName = "0303"
        desDir = join("./log", curExpName)
        print(json.dumps({curExpName:str(kth)}, indent=4))
        setStdoutToDefault(f)

        makeDir(desDir)
        makeAllDir()
        # exit()
    subprocess.call('./train.sh')


if __name__=="__main__":
    brutNas()

    

