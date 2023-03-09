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
    initiManualAssign = {
        "layer_0_1": [
            1,
            0,
            0,
            0,
            0
        ],
        "layer_1_4": [
            1,
            0,
            0,
            0,
            0
        ],
        "layer_4_5": [
            1,
            0,
            0,
            0,
            0
        ],
        # "layer_3_4": [
        #     1,
        #     0,
        #     0,
        #     0,
        #     0
        # ],
        # "layer_4_5": [
        #     1,
        #     0,
        #     0,
        #     0,
        #     0
        # ],
        # "layer_3_5": [
        #     1,
        #     0,
        #     0,
        #     0,
        #     0
        # ],
    }


    curExpName = "0309"
    desDir = join("./log", curExpName)
    makeDir(desDir)
    makeAllDir()
    for kth in range(cfg["numOfKth"]):
        f = setStdoutToFile("./curExperiment.json")
        print(json.dumps({curExpName:str(kth)}, indent=4))
        setStdoutToDefault(f)
        manualAssign = copy.deepcopy(initiManualAssign)
        filePath = "./decode/{}th_decode.json".format(kth)
        f = setStdoutToFile(filePath)
        print(json.dumps(manualAssign, indent=4)) #* make ndarray to list
        setStdoutToDefault(f)   
        # exit()
    subprocess.call('./train.sh')


if __name__=="__main__":
    brutNas()

    

