import torch
from torch.autograd import Variable
import os
import torch.nn as nn
import argparse
from datetime import datetime
# from lib.GMambaPolyp import GMambaPolyp
from lib.model_final_SS2D import GMambaPolyp
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter,create_dir,print_and_save,epoch_time
import torch.nn.functional as F
import numpy as np
import random


device_ids = [0,1]
torch.cuda.set_device(device_ids[0])
torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def test(model, path, dataset):
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
    IOU = 0.0
    with torch.no_grad():
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            # image = image.to(device)
            image = image.to(device_ids[0])
            res, _, _ = model(image)

            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            input = res
            target = np.array(gt)
            N = gt.shape
            smooth = 1
            input_flat = np.reshape(input, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = (input_flat * target_flat)
            dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
            dice = '{:.4f}'.format(dice)
            dice = float(dice)
            union = input_flat + target_flat - intersection
            iou = (intersection.sum() + smooth) / (union.sum() + smooth)
            iou = '{:.4f}'.format(iou)
            iou = float(iou)
            DSC = DSC + dice
            IOU = IOU + iou
    return DSC / num1, IOU / num1



def train(train_loader, model, optimizer, epoch, test_path):

    model.train()
    global best_dice, best_iou, train_log_path, early_stopping_count
    size_rates = [0.75, 1, 1.25]
    loss_P1_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            images = images.to(device_ids[0])

            gts = Variable(gts).cuda()
            gts = gts.to(device_ids[0])


            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate (images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate (gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            P1, P2, P3 = model(images)
            
            if P1.size()!=gts.size():
                P1 = F.interpolate(P1, size=(trainsize, trainsize), mode='bilinear', align_corners=False)
            gts1 = F.interpolate(gts, size=(P2.shape[3], P2.shape[2]), mode='bilinear', align_corners=False)
            gts2 = F.interpolate(gts, size=(P3.shape[3], P3.shape[2]), mode='bilinear', align_corners=False)

            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts1)
            loss_P3 = structure_loss(P3, gts2)
            loss = (loss_P1 + loss_P2 + loss_P3)/3.0

            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss
            if rate == 1:
                loss_P1_record.update(loss_P1.data, opt.batchsize)

        # ---- train visualization ----
        if i % 20 == 0:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P1_record.show()))
        elif i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P1_record.show()))
            data_str = f"P1: {loss_P1}, P2: {loss_P2}, P3: {loss_P3}, total: {loss}, lateral-5: {loss_P1_record.show():0.4f}\n"
            print_and_save(train_log_path, data_str)

    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

#     test
    global dict_plot

    if epoch % 1 == 0:
        dataset1 = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']

        for j in range(0, 5):
            dataset = dataset1[j]
            dataset_dice, dataset_iou = test(model, test_path, dataset)
            data_str = f"{dataset} - Dice:{dataset_dice} - Iou:{dataset_iou}\n"
            print_and_save(train_log_path, data_str)
            if dataset_dice > best_dice[j] and dataset_iou > best_iou[j]:
                best_dice[j] = dataset_dice
                best_iou[j] = dataset_iou
                torch.save(model.state_dict(),
                           save_path + 'train_doublebest_' + dataset + '.pth')
            elif dataset_iou > best_iou[j]:
                best_iou[j] = dataset_iou
                torch.save(model.state_dict(),
                           save_path + 'train_best_iou_' + dataset + '.pth')
            elif dataset_dice > best_dice[j]:
                best_dice[j] = dataset_dice
                torch.save(model.state_dict(),
                           save_path + 'train_best_dice_' + dataset + '.pth')
            data_str = f"\tbest_dice:{best_dice[j]} - best_iou:{best_iou[j]}\n"
            print_and_save(train_log_path, data_str)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    data_str = f"Epoch Time: {epoch_mins}m {epoch_secs}s\n"
    print_and_save(train_log_path, data_str)



if __name__ == '__main__':

    # import pynvml
    import time

    dict_plot = {'CVC-300': [], 'CVC-ClinicDB': [], 'Kvasir': [], 'CVC-ColonDB': [], 'ETIS-LaribPolypDB': [],
                 'test': []}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    model_name = 'train'
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=200, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=24, help='training batch size')
    
    parser.add_argument('--weight_decay', type=int,
                        default=1e-4, help='weight decay')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=30, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='/home/featurize/work/.ssh/PGCF/TrainDatasetEdges',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='/home/featurize/work/.ssh/PGCF/TestDataset',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='/home/featurize/work/.ssh/PGCF/checkpoint_GMambaPolyp/')

    opt = parser.parse_args()


    # ---- build models ----
    model = GMambaPolyp().to(device_ids[0])


    best_dice = [0, 0, 0, 0, 0]
    best_iou = [0, 0, 0, 0, 0]

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=opt.weight_decay, momentum=0.9)


    d = [ 'CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']

    create_dir("logs")
    """ Training logfile """
    train_log_path = "logs/train_logger_model_GMambaPolyp.txt"
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open("logs/train_log.txt", "w")
        train_log.write("\n")
        train_log.close()


    data_str = f"Optimizer: {optimizer}\n"
    print_and_save(train_log_path, data_str)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)
    print(total_step)
    early_stopping_count = 0
    early_stopping_patience = 30
    data_str = f"Early Stopping Patience: {early_stopping_patience}\n"
    print_and_save(train_log_path, data_str)

    """ Record Date & Time """
    datetime_object = str(datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    print("#" * 20, "Start Training", "#" * 20)
    try:
        for epoch in range(0, opt.epoch):
            data_str = f"Epoch: [{epoch+1:03d}/{opt.epoch:03d}]\n"
            print_and_save(train_log_path, data_str)
            start_time = time.time()
            adjust_lr(optimizer, opt.lr, epoch + 1, 0.1, 200)
            train(train_loader, model, optimizer, epoch + 1, opt.test_path)
    except Exception as e:
        print(f"An error occurred: {e}")  # Debugging line
