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
        "layer_0_2": [
            1,
            0,
            0,
            0,
            0
        ],
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

    expNameList = [
    "0327_1",
    "0327_2",
    "0327_3",
    "0327_4",
    "0327_5",
    "0327_6",
    "0327_7",
    "0327_8",
    "0327_9",
    "0327_10",
    "0327_11",
    "0327_12",
    "0327_13",
    "0327_14",
    "0327_15",
    "0327_16",
    "0327_17",
    "0327_18",
    "0327_19",
    "0327_20",
    "0327_21",
    "0327_22",
    "0327_23",
    "0327_24",
    
    ]
    # for curExpName in expNameList:
    curExpName = "0405_11"
    desDir = join("./log", curExpName)
    makeDir(desDir)
    makeAllDir()
    
    f = setStdoutToFile("./curExperiment.json")
    print(json.dumps({curExpName:str(1)}, indent=4))
    setStdoutToDefault(f)
        
    # for kth in range(cfg["numOfKth"]):
    #     f = setStdoutToFile("./curExperiment.json")
    #     print(json.dumps({curExpName:str(kth)}, indent=4))
    #     setStdoutToDefault(f)
    #     manualAssign = copy.deepcopy(initiManualAssign)
    #     filePath = "./decode/{}th_decode.json".format(kth)
    #     f = setStdoutToFile(filePath)
    #     print(json.dumps(manualAssign, indent=4)) #* make ndarray to list
    #     setStdoutToDefault(f)   
        # exit()

            
            
    subprocess.call('./train.sh')


if __name__=="__main__":
    brutNas()

    

