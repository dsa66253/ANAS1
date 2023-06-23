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
        "layer_2_3": [
            1,
            0,
            0,
            0,
            0
        ],
        "layer_3_4": [
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
    }


    expNameList = [
    "0329_1",
    "0329_2",
    "0329_3",
    "0329_4",
    "0329_5",
    "0329_6",
    "0329_7",
    "0329_8",
    "0329_9",
    "0329_10",
    "0329_11",
    "0329_12",
    "0329_13",
    "0329_14",
    "0329_15",
    "0329_16",
    "0329_17",
    "0329_18",
    "0329_19",
    "0329_20",
    "0329_21",
    "0329_22",
    "0329_23",
    "0329_24",
    "0329_25",
    ]
    # for curExpName in expNameList:
    curExpName = "0623_6"
    desDir = join("./log", curExpName)
    makeDir(desDir)
    makeAllDir()
    
    f = setStdoutToFile("./curExperiment.json")
    print(json.dumps({curExpName:str(1)}, indent=4))
    setStdoutToDefault(f)
        
    for kth in range(cfg["numOfKth"]):
        manualAssign = copy.deepcopy(initiManualAssign)
        filePath = "./decode/{}th_decode.json".format(kth)
        f = setStdoutToFile(filePath)
        print(json.dumps(manualAssign, indent=4)) #* make ndarray to list
        setStdoutToDefault(f)   
        # exit()

            
            
    subprocess.call('./train.sh')


if __name__=="__main__":
    brutNas()

    

