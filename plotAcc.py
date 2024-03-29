import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import os
from data.config import cfg_newnasmodel as cfg
def createMAvg(input):
    howMany = 5
    ma = np.copy(input)
    for i in range(0, len(input)):
        window = []
        for j in range(-2, -2+howMany):
            if i+j>=0 and i+j<len(input):
                # print(i+j)
                window.append(input[i+j])
        # print(input)
        ma[i] = np.mean(window)
        # print(i, window, ma[i])
    return ma
def plot_loss_curve(lossRecord, title='default', saveFolder="./"):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    figure(figsize=(6, 4))
    plt.plot(lossRecord['train'], c='tab:red', label='train')
    plt.plot(lossRecord['val'], c='tab:cyan', label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss of {}'.format(title))
    plt.legend()
    
    plt.savefig(os.path.join(saveFolder, title))

def plot_acc_curve(accRecord, title='default', saveFolder="./"):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    fig, ax = plt.subplot()
    ax.plot(accRecord['train'], c='tab:red', label='train')
    ax.plot(accRecord['val'], c='tab:cyan', label='val')
    try:
        ax.plot(accRecord['test'], c='tab:brown', label='test')
    except Exception as e:
        print("null accRecord['test']", e)
    ax.set_xlabel('epoch')
    ax.set_ylabel('acc')
    ax.set_title(format(title))
    ax.legend()
    # plt.show()
    plt.savefig(os.path.join(saveFolder, title)) 
def plot_acc_curves(accRecord, ax, title='default', saveFolder="./"):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    totalEpoch = len(accRecord["train"])
    ax.plot(accRecord['train'], c='tab:red', label='train')
    ax.plot(accRecord['val'], c='tab:cyan', label='val')
    
    try:
        ax.plot(accRecord['test'], c='tab:brown', label='test')
    except Exception as e:
        print("null accRecord['test']", e)
    ax.yaxis.grid()
    ax.xaxis.grid()
    ax.set_yticks(range(0, 110, 10))
    ax.set_xticks(range(0, totalEpoch, 10))
    ax.set_xlabel('epoch')
    ax.set_ylabel('acc')
    ax.set_title(format(title))
    ax.legend()
def plot_combined_acc(folder = "./accLoss", title='combine', saveFolder="./plot", trainType="Nas"):
    numOfAx = 3
    indexOfAx = 0
    numOfFig = cfg["numOfKth"] // numOfAx
    indexOfFig = 0
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    for i in range(numOfFig):
        fig, axs = plt.subplots(numOfAx, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
        for kth in range(numOfAx):
            trainNasTrainAccFile = os.path.join(folder, "{}_train_acc_{}.npy".format(trainType, str(indexOfAx)) )
            trainNasnValAccFile = os.path.join( folder,"{}_val_acc_{}.npy".format(trainType, str(indexOfAx)) )
            testAccFile = os.path.join( folder,"{}_test_acc_{}.npy".format(trainType, str(indexOfAx)) )
            # testAccFile = os.path.join(folder, "trainNasTestAcc_{}.npy".format(trainType, str(kth)) )
            try:
                accRecord = {
                    "train": np.load(trainNasTrainAccFile),
                    "val": np.load(trainNasnValAccFile),
                    "test": np.load(testAccFile)
                }
            except:
                accRecord = {
                    "train": np.load(trainNasTrainAccFile),
                    "val": np.load(trainNasnValAccFile),
                    # "test": np.load(testAccFile)
                }
            plot_acc_curves(accRecord, axs[kth], "acc_"+str(indexOfAx), "./plot")
            indexOfAx = indexOfAx + 1
        indexOfFig = indexOfFig + 1
        fileName = trainType+"_"+  str(indexOfFig)
        print("save png to ", os.path.join(saveFolder, fileName))
        plt.savefig(os.path.join(saveFolder, fileName))

def getAccByMaxValANAS(k, baseDir, expName):
    if expName in ["0322_3"]:
        valAcc = np.load( "./log/{}/accLoss/Nas_val_acc_{}.npy".format(baseDir, str(k))  )
        testAcc = np.load( "./log/{}/accLoss/Nas_test_acc_{}.npy".format(baseDir, str(k))  )
    else:
        valAcc = np.load( "./log/{}/accLoss/retrain_val_acc_{}.npy".format(baseDir, str(k)) )
        testAcc = np.load("./log/{}/accLoss/retrain_test_acc_{}.npy".format(baseDir, str(k)) )
    valIndex = np.argmax(valAcc)
    return round(testAcc[valIndex], 2)
if __name__=="__main__":

    expNameList = ["0322_3", "0322_3_copy"]
    acc = {}
    for expName in expNameList:
        data = []
        for i in range(10):
            data.append(getAccByMaxValANAS(i, expName, expName))
        acc[expName] = data
    ax = plt.subplot()
    for expName in expNameList:
        if expName=="0322_3":
            ax.plot(acc[expName], c='tab:red', label=expName)
        else:
            ax.plot(acc[expName], c='tab:cyan', label=expName)
    ax.set_xlabel('kth')
    ax.set_ylabel('acc')
    # ax.set_title(format(title))
    ax.legend()
    # plt.show()
    plt.savefig("./log/0322_3/acc.png") 
    # plot_combined_acc(trainType="Nas")
    # plot_combined_acc(trainType="retrain")
    # net = "alexnet"
    # folder = "./accLoss" 
    # title='combine_'+net
    # saveFolder="./plot"
    # fig, axs = plt.subplots(1, figsize=(10, 8), sharex=True, constrained_layout=True)
    # for kth in range(1):
    #     trainNasTrainAccFile = os.path.join(folder, "trainNasTrainAcc_{}.npy".format(str(kth)) )
    #     trainNasnValAccFile = os.path.join( folder,"trainNasValAcc_{}.npy".format(str(kth)) )
    #     testAccFile = os.path.join(folder, "trainNasTestAcc_{}.npy".format(str(kth)) )
        
        
    #     accRecord = {
    #         "train": np.load(trainNasTrainAccFile)*100,
    #         "val": np.load(trainNasnValAccFile)*100,
    #         "test": np.load(testAccFile)*100
    #         }
    #     plot_acc_curves(accRecord, axs, "acc_"+str(kth), "./plot")
    # # plt.show()
    # print("save png to ", os.path.join(saveFolder, title))
    # plt.savefig(os.path.join(saveFolder, title))
    exit()
    folder = "./accLoss"
    for kth in range(3):
        trainNasTrainAccFile = os.path.join(folder, "trainNasTrainAcc_{}.npy".format(str(kth)) )
        trainNasnValAccFile = os.path.join( folder,"trainNasValAcc_{}.npy".format(str(kth)) )
        testAccFile = os.path.join(folder, "testAcc_{}.npy".format(str(kth)) )
        
        accRecord = {"train": np.load(trainNasTrainAccFile),
            "val": np.load(trainNasnValAccFile),
            "test": np.load(testAccFile)
            }
        plot_acc_curve(accRecord, "acc_"+str(kth), "./plot")
