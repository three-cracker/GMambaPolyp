import torch
from torch.autograd import Variable
import os
import torch.nn as nn
import argparse
from datetime import datetime
from lib.GMambaPolyp import GMambaPolyp
from utils.utils import CalParams


if __name__ == '__main__':
    model = GMambaPolyp().to("cuda")
    for name, module in model.named_modules():
        print(name, module)
    print(model) 
    input = torch.rand(1,3,352,352).to("cuda")
    CalParams(model,input)

