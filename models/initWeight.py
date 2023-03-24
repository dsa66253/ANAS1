import torch
import torch.nn as nn
import random
import numpy as np
import json, os
def set_seed_cpu(seed):
    # print("set_seed_cpu seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
exp2IniFunc = {
    "0925_5": lambda weight: torch.nn.init.uniform_(weight, -0.025/2, 0.0),
    "0926_5": lambda weight: torch.nn.init.uniform_(weight, 0, 0.025/2),
    "0926_10": lambda weight: torch.nn.init.uniform_(weight, -0.005/2, 0.005/2),
    "0324_11": lambda weight: torch.nn.init.uniform_(weight, -0.1/2, 0.1/2), 
    "0324_12": lambda weight: torch.nn.init.uniform_(weight, -0.05/2, 0.05/2),
    "0324_13": lambda weight: torch.nn.init.uniform_(weight, -0.05/4, 0.05/4),
    "0324_14": lambda weight: torch.nn.init.uniform_(weight, -0.05/8, 0.05/8),
    "0324_15": lambda weight: torch.nn.init.uniform_(weight, -0.05/16, 0.05/16),
    "0324_16": lambda weight: torch.nn.init.normal_(weight, 0, 1/2),
    "0324_17,": lambda weight: torch.nn.init.normal_(weight, 0, 1/4),
    "0324_18": lambda weight: torch.nn.init.normal_(weight, 0, 0.1/2),
    "0324_19": lambda weight: torch.nn.init.normal_(weight, 0, 0.1/4),
    "0324_19": lambda weight: torch.nn.init.normal_(weight, 0, 0.01/2),
    "0324_20": lambda weight: torch.nn.init.kaiming_normal_(weight),
    "0324_21": lambda weight: torch.nn.init.uniform_(weight, -1, 1),
    "0324_22": lambda weight: torch.nn.init.uniform_(weight, -1./2, 1./2),
    "0324_23": lambda weight: torch.nn.init.uniform_(weight, -0.25, 0.25),
    "0324_24": lambda weight: torch.nn.init.uniform_(weight, -0.25/2, 0.25/2),
}
def openCurExp():
    filePath = os.path.join("./curExperiment.json")
    f = open(filePath)
    exp = json.load(f)
    for key in exp:
        return key 
    #! here need to be related to kth 
def initialize_weights(model, seed):
    print("set initialize weight with seed ", seed)
    curExp = openCurExp()
    print("cuurent experiment", curExp)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            set_seed_cpu(seed)
            exp2IniFunc[curExp](m.weight)
            # torch.nn.init.kaiming_normal_(m.weight)
            # m.weight = torch.abs(m.weight)
            # torch.nn.init.uniform_(m.weight, -0.005/2, 0.005/2)
            # m.weight.data.fill_(0)
            # setTensorPositive(m.weight)
            # torch.nn.init.normal_(m.weight, 0.025, 0.025/2)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 1)
        elif isinstance(m, nn.Linear):
            set_seed_cpu(seed)
            exp2IniFunc[curExp](m.weight)
            # torch.nn.init.kaiming_normal_(m.weight)
            # setTensorPositive(m.weight.data)
            # torch.nn.init.uniform_(m.weight, -0.005/2, 0.005/2)
            # m.weight.data.fill_(0)
            # nn.init.constant_(m.bias, 0)
            # torch.nn.init.normal_(m.weight, 0.025, 0.025/2)
            pass
        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                m.weight.data.fill_(1)
                m.bias.data.zero_()
def setTensorPositive(tensor):
    tmp = torch.zeros(tensor.shape)
    tmp = torch.nn.init.kaiming_normal_(tmp)
    tmp = torch.abs(tmp)
    with torch.no_grad():
        tensor*=0
        tensor+= tmp
