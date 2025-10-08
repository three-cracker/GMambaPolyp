import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
from lib.model_final import GMambaPolyp
from utils.dataloader import test_dataset
import torch.nn as nn


device_ids = [0,1]
device = torch.device('cuda:0')

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
# path = '/home/featurize/work/.ssh/PGCF/checkpoint_dw+scsa/train_doublebest_CVC-300.pth'
dataset = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']
# dataset = ['CVC-ColonDB', 'Kvasir']
# dataset = ['CVC-ClinicDB']
# dataset = ['Kvasir']


def find_files_with_prefix(directory, prefix):  
    matching_files = []  
    # 遍历指定目录  
    for filename in os.listdir(directory):  
        if filename.startswith(prefix):  
            matching_files.append(filename)  
    return matching_files  



for _data_name in dataset:
    path = '/home/featurize/work/.ssh/PGCF/checkpoint_MambaPolypNet/train_doublebest_{}.pth'.format(_data_name)
    # path = '/home/featurize/work/.ssh/PGCF/checkpoint_pranet/PraNet-19.pth'.format(_data_name)
    data_path = '/home/featurize/work/.ssh/PGCF/TestDataset/{}/'.format(_data_name)
    save_path = '/home/featurize/work/.ssh/PGCF/result_GMambaPolyp/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = GMambaPolyp().to(device_ids[0])
    # model = PraNet().to(device_ids[0])
    # model = nn.DataParallel(model, device_ids=device_ids).to(device_ids[0])
    # model.load_state_dict(torch.load(path,map_location=device))

    save_model = torch.load(path,map_location=device)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res,_,_ = model(image)
        # res,_,_,_ = model(image)
        # res = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path+name, res*255)