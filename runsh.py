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
        "layer_1_2": [
            1,
            0,
            0,
            0,
            0
        ],
        # "layer_2_3": [
        #     1,
        #     0,
        #     0,
        #     0,
        #     0
        # ],
        "layer_2_4": [
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
        # "layer_3_5": [
        #     1,
        #     0,
        #     0,
        #     0,
        #     0
        # ],
    }

    expNameList = [
    "0328_1",
    "0328_2",
    "0328_3",
    "0328_4",
    "0328_5",
    "0328_6",
    "0328_7",
    "0328_8",
    "0328_9",
    "0328_10",
    "0328_11",
    "0328_12",
    "0328_13",
    "0328_14",
    "0328_15",
    "0328_16",
    "0328_17",
    "0328_18",
    "0328_19",
    "0328_20",
    "0328_21",
    "0328_22",
    "0328_23",
    "0328_24",
    ]
    # for curExpName in expNameList:
    curExpName = "0405_8"
    desDir = join("./log", curExpName)
    makeDir(desDir)
    makeAllDir()
    
    f = setStdoutToFile("./curExperiment.json")
    print(json.dumps({curExpName:str(1)}, indent=4))
    setStdoutToDefault(f)
        
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

    

