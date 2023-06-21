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
    "0328_1": lambda weight: torch.nn.init.uniform_(weight, -0.5, 0.0),
    "0328_2": lambda weight: torch.nn.init.uniform_(weight, 0, 0.5),
    "0328_3": lambda weight: torch.nn.init.uniform_(weight, -1, 0),
    "0328_4": lambda weight: torch.nn.init.uniform_(weight, 0, 1),
    "0328_5": lambda weight: torch.nn.init.uniform_(weight, -2, 2),
    "0328_6": lambda weight: torch.nn.init.uniform_(weight, -1, 1),
    "0328_7": lambda weight: torch.nn.init.uniform_(weight, -1./2, 1./2),
    "0328_8": lambda weight: torch.nn.init.uniform_(weight, -0.25, 0.25),
    "0328_9": lambda weight: torch.nn.init.uniform_(weight, -0.25/2, 0.25/2),
    "0328_10": lambda weight: torch.nn.init.uniform_(weight, -0.1/2, 0.1/2), 
    "0328_11": lambda weight: torch.nn.init.uniform_(weight, -0.05/2, 0.05/2),
    "0328_12": lambda weight: torch.nn.init.uniform_(weight, -0.05/4, 0.05/4),
    "0328_13": lambda weight: torch.nn.init.uniform_(weight, -0.05/8, 0.05/8),
    "0328_14": lambda weight: torch.nn.init.uniform_(weight, -0.05/16, 0.05/16),
    "0328_15": lambda weight: torch.nn.init.normal_(weight, 0, 1/2),
    "0328_16": lambda weight: torch.nn.init.normal_(weight, 1/2, 1/2),
    "0328_17": lambda weight: torch.nn.init.normal_(weight, -1/2, 1/2),
    "0328_18": lambda weight: torch.nn.init.normal_(weight, 0, 1/4),
    "0328_19": lambda weight: torch.nn.init.normal_(weight, 0, 0.1/2),
    "0328_20": lambda weight: torch.nn.init.normal_(weight, 0, 0.1/4),
    "0328_21": lambda weight: torch.nn.init.normal_(weight, 0, 0.01/2),
    "0328_22": lambda weight: torch.nn.init.normal_(weight, 0, 0.01/4),
    "0328_23": lambda weight: torch.nn.init.normal_(weight, 0, 0.001/2),
    "0328_24": lambda weight: torch.nn.init.kaiming_normal_(weight),
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
            exp2IniFunc["0329_11"](m.weight)
            # torch.nn.init.kaiming_normal_(m.weight)
            # m.weight = torch.abs(m.weight)
            # torch.nn.init.uniform_(m.weight, -0.05/4, 0.05/4),
            # m.weight.data.fill_(0)
            # setTensorPositive(m.weight)
            # torch.nn.init.normal_(m.weight, 0.025, 0.025/2)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 1)
        elif isinstance(m, nn.Linear):
            set_seed_cpu(seed)
            exp2IniFunc["0329_11"](m.weight)
            # torch.nn.init.kaiming_normal_(m.weight)
            # setTensorPositive(m.weight.data)
            # torch.nn.init.uniform_(m.weight, -0.05/4, 0.05/4),
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
