import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import torchvision
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
from tensorboardX import SummaryWriter

from model.deeplab_multi import DeeplabMulti
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet

import evaluate_cityscapes
import compute_iou

import time

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = '../dataset/gta5/'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'
DATA_DIRECTORY_TARGET = '../dataset/Cityscapes/leftImg8bit_trainvaltest/'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
INPUT_SIZE_TARGET = '1024,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 150000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
LOG_DIR = './log'

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET = 0.001
LAMBDA_ADV_LOCAL = 0.0002
LAMBDA_SEG_LOCAL = 0.1
LAMBDA_MATCH_TARGET1 = 0.0002
LAMBDA_MATCH_TARGET2 = 0.001

TARGET = 'cityscapes'
SET = 'train'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target", type=float, default=LAMBDA_ADV_TARGET,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-match-target1", type=float, default=LAMBDA_MATCH_TARGET1,
                        help="lambda_match for patch match.")
    parser.add_argument("--lambda-match-target2", type=float, default=LAMBDA_MATCH_TARGET2,
                        help="lambda_match for patch match.")
    parser.add_argument("--lambda-adv-local", type=float, default=LAMBDA_ADV_LOCAL,
                        help="lambda_adv for local target.")
    parser.add_argument("--lambda-seg-local", type=float, default=LAMBDA_SEG_LOCAL,
                        help="lambda_adv for local target.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--tensorboard", action='store_true', help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    return parser.parse_args()


args = get_arguments()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def get_feature(vgg19, img_tensor, feature_id):
    feature_tensor = vgg19.features[:feature_id](img_tensor)
    return feature_tensor

def normalize(feature):
    f_norm = torch.norm(feature, dim=1, keepdim=True).detach()
    f_norm[torch.abs(f_norm) < 1e-6] = 1
    return feature / f_norm

def get_correspondance(f_src, f_tar, pred, pred_target, prob_threshold = 0.95, sampling_num = 100):
    flatten = nn.Flatten(2,-1)
    bn,f_dim,hs,ws = f_src.size()
    _, _, ht, wt = f_tar.size()
    interp_fs = nn.Upsample(size=(hs, ws), mode='bilinear', align_corners=True)
    interp_ft = nn.Upsample(size=(ht, wt), mode='bilinear', align_corners=True)
    src_mask = (F.softmax(interp_fs(pred), dim=1) > prob_threshold).detach()
    tar_mask = (F.softmax(interp_ft(pred_target), dim=1) > prob_threshold).detach()
    s_mask = torch.zeros(bn, 1, hs, ws).to(f_src.get_device()).detach()
    t_mask = torch.zeros(bn, 1, ht, wt).to(f_tar.get_device()).detach()
    for k in range(args.num_classes):
        s_mask[src_mask[:,k:k+1,:,:]] = 1
        t_mask[tar_mask[:,k:k+1,:,:]] = 1  
    f_src = flatten(f_src * s_mask)#[bsize, channels, w*h]
    f_tar = flatten(f_tar * t_mask)
    del s_mask, t_mask, src_mask, tar_mask
    co_relation = f_src.squeeze(0).transpose(0,1).mm(f_tar.squeeze(0)).detach().to('cpu')
    relate_id = (co_relation > prob_threshold).nonzero()
    if relate_id.nelement()>sampling_num*2:
        idx = torch.randperm(sampling_num)
        relate_id = relate_id[idx,:]
    return relate_id

def local_feature_loss(corres_id, s_label, f_tar, model, seg_loss):
    _, _, ht, wt = f_tar.size()
    interp_ft = nn.Upsample(size=(ht, wt), mode='nearest')
    flatten = nn.Flatten(2,-1)
    if corres_id.nelement()>0:
        s_label = interp_ft(s_label.unsqueeze(1).to(torch.float))
        s_label = flatten(s_label)[:, :, corres_id[:,0]].unsqueeze(3)
        f_tar_in = flatten(f_tar)[:, :, corres_id[:,1]].unsqueeze(3)
        tar_out = model.classifier(f_tar_in)
        return seg_loss(tar_out, s_label.long().squeeze(1))
    else:
        return 0

def main():
    """Create the model and start the training."""

    device = torch.device("cuda" if not args.cpu else "cpu")

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True

    # Create network
    if args.model == 'DeepLab':
        model = DeeplabMulti(num_classes=args.num_classes)
        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)
        #model.load_state_dict(saved_state_dict)

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if not args.num_classes == 19 or not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                # print i_parts
        model.load_state_dict(new_params)

    model.train()
    model.to(device)

    cudnn.benchmark = True

    # init D
    model_D1 = FCDiscriminator(num_classes=args.num_classes).to(device)
    model_D2 = FCDiscriminator(num_classes=args.num_classes).to(device)
    
#     model_D1.load_state_dict(torch.load('./snapshots/local_00002/GTA5_21000_D1.pth'))
#     model_D2.load_state_dict(torch.load('./snapshots/local_00002/GTA5_21000_D2.pth'))

    model_D1.train()
    model_D1.to(device)
    
    model_D2.train()
    model_D2.to(device)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     crop_size=input_size_target,
                                                     scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                     set=args.set),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)


    targetloader_iter = enumerate(targetloader)

    # Load VGG
    #vgg19 = torchvision.models.vgg19(pretrained=True)
    #vgg19.to(device)

    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()
    
    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1

    # set up tensor board
    if args.tensorboard:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        writer = SummaryWriter(args.log_dir)

    for i_iter in range(0, args.num_steps):

        loss_seg_value = 0
#         loss_seg_local_value = 0
        loss_adv_target_value = 0
#         loss_adv_local_value = 0
        loss_D_value = 0
#         loss_D_local_value = 0
        loss_local_match_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D1.zero_grad()
        adjust_learning_rate_D(optimizer_D1, i_iter)
        
        optimizer_D2.zero_grad()
        adjust_learning_rate_D(optimizer_D2, i_iter)

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False
                
            for param in model_D2.parameters():
                param.requires_grad = False

            # train with source

            _, batch = trainloader_iter.__next__()

            images, labels, _, _ = batch
            images = images.to(device)
            labels = labels.long().to(device)

            pred_s1, pred_s2, _, _ = model(images)
            #f_s2 = normalize(f_s2)
            pred_s1, pred_s2 = interp(pred_s1), interp(pred_s2)

            loss_seg = args.lambda_seg * seg_loss(pred_s1, labels) + seg_loss(pred_s2, labels)
            del labels

            # proper normalization
            loss_seg_value += loss_seg.item() / args.iter_size

            # train with target
            _, batch = targetloader_iter.__next__()
            images, _, _ = batch
            images = images.to(device)

            pred_t1, pred_t2, _, _ = model(images)
            #f_t2 = normalize(f_t2)
            pred_t1, pred_t2 = interp_target(pred_t1), interp_target(pred_t2)
            del images

            D_out_1 = model_D1(F.softmax(pred_t1, dim=1))
            D_out_2 = model_D2(F.softmax(pred_t2, dim=1))

            loss_adv_target1 = bce_loss(D_out_1, torch.FloatTensor(D_out_1.data.size()).fill_(source_label).to(device))
            loss_adv_target2 = bce_loss(D_out_2, torch.FloatTensor(D_out_2.data.size()).fill_(source_label).to(device))
            loss_adv_target_value += (args.lambda_adv_target1 * loss_adv_target1 + args.lambda_adv_target * loss_adv_target2).item() / args.iter_size
            
            loss = loss_seg + args.lambda_adv_target1 * loss_adv_target1 + args.lambda_adv_target * loss_adv_target2

            del D_out_1, D_out_2

#             #< Local patch part>#
#             corres_id2 = get_correspondance(f_s2, f_t2, pred_s2, pred_t2)
                
#             #loss_local1 = local_feature_loss(corres_id1, f_s1, f_t1, model, seg_loss)
#             loss_local2 = local_feature_loss(corres_id2, labels, f_t2, model, seg_loss)
#             loss_local = args.lambda_match_target2 * loss_local2 #+args.lambda_match_target1 * loss_local1
#             loss += loss_local
#             if corres_id2.nelement() > 0:
#                 loss_local_match_value += loss_local.item()/ args.iter_size
            
            loss /= args.iter_size
            loss.backward()


            # train D

            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True
            for param in model_D2.parameters():
                param.requires_grad = True


            # train with source
            pred_s1, pred_s2 = pred_s1.detach(), pred_s2.detach()
            D_out_1, D_out_2 = model_D1(F.softmax(pred_s1)), model_D2(F.softmax(pred_s2))
            loss_D_1 = bce_loss(D_out_1, torch.FloatTensor(D_out_1.data.size()).fill_(source_label).to(device))/ args.iter_size / 2
            loss_D_2 = bce_loss(D_out_2, torch.FloatTensor(D_out_2.data.size()).fill_(source_label).to(device))/ args.iter_size / 2
            loss_D_1.backward()
            loss_D_2.backward()

            loss_D_value += (loss_D_1+loss_D_2).item()

            # train with target
            pred_t1, pred_t2 = pred_t1.detach(), pred_t2.detach()
            D_out_1, D_out_2 = model_D1(F.softmax(pred_t1)), model_D2(F.softmax(pred_t2))
            loss_D_1 = bce_loss(D_out_1, torch.FloatTensor(D_out_1.data.size()).fill_(target_label).to(device))/ args.iter_size / 2
            loss_D_2 = bce_loss(D_out_2, torch.FloatTensor(D_out_2.data.size()).fill_(target_label).to(device))/ args.iter_size / 2
            loss_D_1.backward()
            loss_D_2.backward()

            loss_D_value += (loss_D_1+loss_D_2).item()
        
        optimizer.step()
        optimizer_D1.step()
        optimizer_D2.step()
        
        if i_iter % 1000 == 0:
            val_dir = '../dataset/Cityscapes/leftImg8bit_trainvaltest/'
            val_list = './dataset/cityscapes_list/val.txt'
            save_dir = './results/tmp'
            gt_dir = '../dataset/Cityscapes/gtFine_trainvaltest/gtFine/val'
            evaluate_cityscapes.test_model(model, device, val_dir, val_list, save_dir)
            mIoU = compute_iou.mIoUforTest(gt_dir, save_dir)

        if args.tensorboard:
            scalar_info = {
                'loss_seg': loss_seg_value,
                #'loss_seg_local': loss_seg_local_value,
                'loss_adv_target': loss_adv_target_value,
                'loss_local_match': loss_local_match_value,
                'loss_D': loss_D_value,
                'mIoU' : mIoU
                #'loss_D_local': loss_D_local_value
            }

            if i_iter % 10 == 0:
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, i_iter)

        print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {0:8d}/{1:8d}, loss_seg = {2:.3f} loss_adv = {3:.3f}, loss_D = {4:.3f}, loss_local_match = {5:.3f}, mIoU = {6:3f} '
        #'loss_seg_local = {5:.3f} loss_adv_local = {6:.3f}, loss_D_local = {7:.3f}'
            .format(i_iter, args.num_steps, loss_seg_value, loss_adv_target_value, loss_D_value, loss_local_match_value, mIoU)
                    #loss_seg_local_value, loss_adv_local_value, loss_D_local_value)
        )
        

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D1.pth'))
            torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D2.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D1.pth'))
            torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D2.pth'))

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()
