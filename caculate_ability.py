import torch
from torch.autograd import Variable
import os
import torch.nn as nn
import argparse
from datetime import datetime
# from lib.model_vmamba4 import VMamba_PGCF as PGCFv2
# from lib.model_vamamba_SS2D import PGCF as PGCFv2
from lib.model_PGCF import PGCF as PGCFv1
from lib.model_final_SS2D import GMambaPolyp as PGCFv2
# from test import model
from utils.utils import CalParams


if __name__ == '__main__':
    model1 = PGCFv1().to("cuda")
    model2 = PGCFv2().to("cuda")
    for name, module in model2.named_modules():
        print(name, module)
    print(model2) 
    input = torch.rand(1,3,352,352).to("cuda")
    # model2.train()
    # model2(input)
    CalParams(model1,input)
    print("*"*60)
    CalParams(model2,input)
