import os
import json
import torch
import argparse
import torch.optim as optim
from test import TestController
import torch.nn as nn
import math
import time
from torchvision import datasets
import torchvision.transforms as T
from data.config import cfg_nasmodel as cfg, cfg_alexnet, trainDataSetFolder, seed
from alexnet.alexnet import Baseline
# from tensorboardX import SummaryWriter #* how about use tensorbaord instead of tensorboardX
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from data.config import folder
from feature.normalize import normalize
from feature.make_dir import makeDir
from feature.split_data import split_data
from feature.random_seed import set_seed_cpu
from PIL import ImageFile
from tqdm import tqdm
from models.mymodel import Model
from data.config import epoch_to_drop
from feature.utility import getCurrentTime, setStdoutToDefault, setStdoutToFile, accelerateByGpuAlgo
from feature.utility import plot_acc_curve, plot_loss_curve, get_device
from utility.alphasMonitor import AlphasMonitor
# from utility.TransformImgTester import TransformImgTester
from utility.DatasetHandler import DatasetHandler
from utility.HistDrawer import HistDrawer
from  utility.DatasetReviewer import DatasetReviewer
from utility.AccLossMonitor import AccLossMonitor
from utility.BetaMonitor import BetaMonitor
from models.initWeight import initialize_weights
from utility.ValController import ValController
stdoutTofile = True
accelerateButUndetermine = False
recover = False

def parse_args(k=0):
    parser = argparse.ArgumentParser(description='imagenet nas Training')
    parser.add_argument('--network', default='nasmodel', help='Backbone network alexnet or nasmodel')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--nas_lr', '--nas-learning-rate', default=3e-3, type=float,
                        help='initial learning rate for nas optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--resume_net', default=None, help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--trainDataSetFolder', default='../dataset/train',
                    help='training data set folder')
    args = parser.parse_args()
    return args

def prepareDataSet():
    #info prepare dataset
    datasetHandler = DatasetHandler(trainDataSetFolder, cfg, seed_weight)
    # datasetHandler.addAugmentDataset(transforms.RandomHorizontalFlip(p=1))
    # datasetHandler.addAugmentDataset(transforms.RandomRotation(degrees=10))
    print("dataset:", trainDataSetFolder)
    print("training dataset set size:", len(datasetHandler.getTrainDataset()))
    print("val dataset set size:", len(datasetHandler.getValDataset()))
    print("class_to_idx", datasetHandler.getClassToIndex())
    return datasetHandler.getTrainDataset(), datasetHandler.getValDataset()

def prepareDataLoader(trainData, valData):
    #info prepare dataloader
    train_loader = torch.utils.data.DataLoader(trainData, batch_size=batch_size, num_workers=args.num_workers,
                                            shuffle=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(valData, batch_size=batch_size, num_workers=args.num_workers,
                                            shuffle=False, pin_memory=True)
    return train_loader, val_loader

def prepareLossFunction():
    print('Preparing loss function...')
    return  nn.CrossEntropyLoss()

def prepareModel(kth):
    #info load full layer json
    filePath = "./data/fullLayer.json"
    f = open(filePath)
    archDict = json.load(f)
    #info prepare model
    print("Preparing model...")
    if cfg['name'] == 'alexnet':
        # alexnet model
        net = Baseline(cfg["numOfClasses"])
        net = net.to(device)
        net.train()
    elif cfg['name'] == 'NasModel':
        # nas model
        # todo why pass no parameter list to model, and we got cfg directly in model.py from config.py
        net = Model(arch=archDict)
        print("net", net)
        #! move to cuda before assign net's parameters to optim, otherwise, net on cpu will slow down training speed
        net = net.to(device)
        net.train()
    initialize_weights(net, seed_weight)
    return net

def prepareOpt(net):
    #info prepare optimizer
    print("Preparing optimizer...")
    if cfg['name'] == 'alexnet':  # BASELINE
        optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
        return optimizer
    elif cfg['name'] == 'NasModel':
        model_optimizer = optim.SGD(net.getWeight(), lr=initial_lr, momentum=momentum,
                                    weight_decay=weight_decay)
        nas_optimizer = optim.Adam(net.getAlphasPara(), lr=nas_initial_lr, weight_decay=weight_decay)
        beta_optimizer = optim.Adam(net.getBetaPara(), lr=nas_initial_lr, weight_decay=weight_decay)
        return model_optimizer, nas_optimizer, beta_optimizer
    
def printNetWeight(net):
    for name, para in net.named_parameters():
        print(name, para)


    
def saveCheckPoint(kth, epoch, optimizer, net, lossRecord, accReocrd):
    print("save check point kth {} epoch {}".format(kth, epoch))
    if epoch==0:
        return 
    # print("net.state_dict()", net.state_dict())
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': lossRecord,
            'acc': accReocrd
            }, 
            os.path.join(folder["savedCheckPoint"], "{}_{}_{}.pt".format(args.network, kth, epoch)))
    except Exception as e:
        print("Failt to save check point")
        print(e)
def recoverFromCheckPoint(kth, epoch, model, optimizer):
    checkpoint = torch.load(os.path.join(folder.savedCheckPoint, "{}_{}_{}.pt".format(args.network, kth, epoch)))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("recover from check point at epoch ", checkpoint['epoch'])
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']

def saveAccLoss(kth, lossRecord, accRecord):
    print("save record to ", folder["accLossDir"])
    try:
        np.save(os.path.join(folder["accLossDir"], "trainNasTrainLoss_"+str(kth)), lossRecord["train"])
        np.save(os.path.join(folder["accLossDir"], "trainNasValnLoss_"+str(kth)), lossRecord["val"])
        np.save(os.path.join(folder["accLossDir"], "trainNasTrainAcc_"+str(kth)), accRecord["train"])
        np.save(os.path.join(folder["accLossDir"], "trainNasValAcc_"+str(kth)), accRecord["val"])
        np.save(os.path.join(folder["accLossDir"], "trainNasTestAcc_"+str(kth)), accRecord["test"])
    except Exception as e:
        print("Fail to save acc and loss")
        print(e)
def makeAllDir():
    for folderName in folder:
        # print("making folder ", folder[folderName])
        makeDir(folder[folderName])

def weightCount(net):
    count = 0
    for netLayerName , netLyaerPara in net.named_parameters():
        print(netLyaerPara.device)
        shape = netLyaerPara.shape
        dim=1
        for e in shape:
            dim = e*dim
        count = count + dim
    return count
def gradCount(net):
    count = 0
    for netLayerName , netLyaerPara in net.named_parameters():
        if netLyaerPara.grad!=None:
            shape = netLyaerPara.grad.shape
            dim=1
            for e in shape:
                dim = e*dim
            count = count + dim
    return count
        
        
def myTrain(kth, trainData, train_loader, val_loader, net, model_optimizer, nas_optimizer, criterion, writer, beta_optimizer):
    
    # other setting
    print("start to train...")
    
    record_train_loss = np.array([])
    record_val_loss = np.array([])
    record_train_acc = np.array([])
    record_val_acc = np.array([])
    record_test_acc = np.array([])
    alphaMonitor = AlphasMonitor()
    betaMonitor = BetaMonitor()
    for epoch in tqdm(range(cfg["epoch"]), unit =" iter on {}".format(kth)):
        print("start epoch", epoch)
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        net.train() # set the model to training mode
        if epoch >= cfg['start_train_nas_epoch'] and (epoch - cfg['start_train_nas_epoch'])%2 == 0:
            #info train alpha and beta
            for i, data in enumerate(trainDataLoader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                nas_optimizer.zero_grad(set_to_none=True)
                beta_optimizer.zero_grad(set_to_none=True)
                outputs = net(inputs) 

                batch_loss = criterion(outputs, labels)
                _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
                batch_loss.backward() 
                nas_optimizer.step() 
                beta_optimizer.step() 

                train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
                train_loss += batch_loss.item()
        else:
            # info train weight
            for i, data in enumerate(trainDataLoader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                model_optimizer.zero_grad() 
                outputs = net(inputs) 

                batch_loss = criterion(outputs, labels)
                _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
                batch_loss.backward() 
                model_optimizer.step() 

                train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
                train_loss += batch_loss.item()
        #info log alpha and beta
        alphaMonitor.logAlphaDictPerIter(net)
        betaMonitor.logBetaDictPerIter(net)
            
        #info handle alpha operation 
        if epoch in epoch_to_drop:
            # pass
            # net.dropMinAlpha()
            # net.dropMinBeta()
            model_optimizer, nas_optimizer, beta_optimizer = prepareOpt(net)
        if epoch >= cfg['start_train_nas_epoch']:
            # if net.dropBeta():
            #     model_optimizer, nas_optimizer, beta_optimizer = prepareOpt(net)
            # net.filtAlphas()
            # net.normalizeAlphas()
            # net.normalizeByDivideSum()
            # net.saveMask(epoch, kth)
            pass
    
        record_train_acc = np.append(record_train_acc, train_acc/len(trainData)*100)
        record_train_loss = np.append(record_train_loss, train_loss/len(trainDataLoader))
        #info validate model
        valAcc = valC.val(net)
        record_val_acc = np.append(record_val_acc, valAcc)
        record_val_loss = np.append(record_val_loss, torch.Tensor([0]))
        # print("epoch", epoch)
        # print("record_val_acc", record_val_acc)
        # print("record_train_acc", record_train_acc)
        #info test model
        testAcc = testC.test(net)
        record_test_acc = np.append(record_test_acc, testAcc)
    
    #info save alpha and beta
    alphaMonitor.saveAllAlphas(kth)
    # alphaMonitor.saveAllAlphasGrad(kth) #! a bug didn't be solved
    betaMonitor.saveAllBeta(kth)
    # betaMonitor.saveAllBetaGrad(kth)
    
    last_epoch_val_acc = valC.val(net)
    lossRecord = {"train": record_train_loss, "val": record_val_loss}
    accRecord = {"train": record_train_acc, "val": record_val_acc, "test": record_test_acc}
    print("start test model before save model")
    testAcc = testC.test(net)
    # print(record_val_acc)
    # print(record_train_acc)
    # testC.printAllModule(net)
    torch.save(net.state_dict(), os.path.join(folder["retrainSavedModel"], cfg['name'] + str(kth) + '_Final.pt'))
    
    return last_epoch_val_acc, lossRecord, accRecord
    
if __name__ == '__main__':
    device = get_device()
    torch.device(device)
    print("running on device: {}".format(device))
    torch.set_printoptions(precision=6, sci_mode=False, threshold=1000)
    torch.set_default_dtype(torch.float32) #* torch.float will slow the training speed
    valList = []
    
    for k in range(0, cfg["numOfKth"]):
        #info set stdout to file
        if stdoutTofile:
            f = setStdoutToFile( os.path.join( folder["log"], "train_nas_5cell_{}th.txt".format(str(k)) ) )
        print("working directory ", os.getcwd())

        #info set seed
        seed_weight = seed[str(k)]
        accelerateByGpuAlgo(cfg["cuddbenchMark"])
        set_seed_cpu(seed_weight)  # 控制照片批次順序
        print("seed_weight{} start at ".format(seed_weight), getCurrentTime())
        print("cfg", cfg)
        args = parse_args(str(k))
        
        makeAllDir()

        img_dim = cfg['image_size']
        num_gpu = cfg['ngpu']
        batch_size = cfg['batch_size']
        gpu_train = cfg['gpu_train']

        num_workers = args.num_workers
        momentum = args.momentum
        weight_decay = args.weight_decay
        initial_lr = args.lr
        nas_initial_lr = args.nas_lr
        gamma = args.gamma
        #info training process 
        trainData, valData = prepareDataSet()
        trainDataLoader, valDataLoader = prepareDataLoader(trainData, valData)
        
        criterion = prepareLossFunction()
        net = prepareModel(k)
        histDrawer = HistDrawer(folder["pltSavedDir"])
        histDrawer.drawNetConvWeight(net, tag="NAS_ori_{}".format(str(k)))
        #info test
        testC = TestController(cfg, device)
        writer = SummaryWriter(log_dir=folder["tensorboard_trainNas"], comment="{}th".format(str(k)))
        #info validation controller
        valC = ValController(cfg, device, valDataLoader, criterion)
        # transformImgTest = TransformImgTester(batch_size, kth=k)



        model_optimizer, nas_optimizer, beta_optimizer = prepareOpt(net)
        last_epoch_val_ac, lossRecord, accRecord  = myTrain(k, trainData, trainDataLoader, valDataLoader, net, model_optimizer, nas_optimizer, criterion, writer, beta_optimizer)  # 進入model訓練
        #info record training processs
        alMonitor = AccLossMonitor(k, folder["pltSavedDir"], folder["accLossDir"], trainType="Nas")
        alMonitor.plotAccLineChart(accRecord)
        alMonitor.plotLossLineChart(lossRecord)
        alMonitor.saveAccLossNp(accRecord, lossRecord)
        valList.append(last_epoch_val_ac)
        print('train validate accuracy:', valList)
        
        datasetReviewer = DatasetReviewer(cfg["batch_size"],
                                        k,
                                        DatasetHandler.getOriginalDataset(trainDataSetFolder, cfg, seed_weight), 
                                        device)
        datasetReviewer.makeSummary(trainDataLoader, writer, net)
        datasetReviewer.showReport()
        
        writer.close()
        if stdoutTofile:
            setStdoutToDefault(f)
        # exit()/ #* for examine why same initial value will get different trained model
    print('train validate accuracy:', valList)
        
        
        
        
        
        
        
        
        
        
        