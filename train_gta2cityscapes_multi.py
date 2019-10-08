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

from utils import bds_voting
from utils import matching_loss

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
SAVE_PRED_EVERY = 500 #5000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
LOG_DIR = './log'

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
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
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
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
    model_D = FCDiscriminator(num_classes=args.num_classes).to(device)
    
#     model_D1.load_state_dict(torch.load('./snapshots/local_00002/GTA5_21000_D1.pth'))
#     model_D2.load_state_dict(torch.load('./snapshots/local_00002/GTA5_21000_D2.pth'))

    model_D.train()
    model_D.to(device)

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

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

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

    for i_iter in range(21000, args.num_steps):

        loss_seg_value = 0
        loss_seg_local_value = 0
        loss_adv_target_value = 0
        loss_adv_local_value = 0
        loss_D_value = 0
        loss_D_local_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D.parameters():
                param.requires_grad = False

            # train with source

            _, batch = trainloader_iter.__next__()

            images, labels, _, _ = batch
            images = images.to(device)
            labels = labels.long().to(device)

            _, pred = model(images)
            source_feature = pred / torch.norm(pred, dim=1, keepdim=True)
            source_feature = source_feature.detach()
            pred = interp(pred)

            loss_seg = seg_loss(pred, labels)

            # proper normalization
            loss_seg_value += loss_seg.item() / args.iter_size

            # train with target
            _, batch = targetloader_iter.__next__()
            images, _, _ = batch
            images = images.to(device)

            _, pred_target = model(images)
            target_feature = pred_target / torch.norm(pred_target, dim=1, keepdim=True)
            target_feature = target_feature.detach()
            pred_target = interp_target(pred_target)

            D_out = model_D(F.softmax(pred_target))

            loss_adv_target = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))

            loss = loss_seg + args.lambda_adv_target1 * loss_adv_target
            loss /= args.iter_size
            loss.backward(retain_graph=True)

            loss_adv_target_value += loss_adv_target.item() / args.iter_size

            # build guidance seg feature map
#             S2T_nnf = bds_voting.PatchMatch(source_feature, target_feature).forward()
#             T2S_nnf = bds_voting.PatchMatch(target_feature, source_feature).forward()
            
#             class_p = bds_voting.build_prob_map(labels.unsqueeze(0), args.num_classes)
#             g = bds_voting.bds_vote(class_p, S2T_nnf, T2S_nnf)
#             g_feature = bds_voting.bds_vote(source_feature, S2T_nnf, T2S_nnf).to(device).detach()
            
#             g = g.to(device).detach()
#             g_feature = g_feature.to(device).detach()
#             g_feature /= torch.norm(g_feature, dim=1, keepdim=True)
#             _, _, gh, gw = g.size()

#             pred_targ_match1 = F.upsample(pred_target1, size=(gh, gw), mode='bilinear')
#             pred_targ_match2 = F.upsample(pred_target2, size=(gh, gw), mode='bilinear')

#             softmax = torch.nn.Softmax().to(device)
#             mloss = matching_loss.compute_matching_loss

#             loss_matching_value1 = mloss(pred_targ_match1, g, target_feature, g_feature, softmax).item() / args.iter_size
#             loss_matching_value2 = mloss(pred_targ_match2, g, target_feature, g_feature, softmax).item() / args.iter_size

#             loss = args.lambda_match_target1 * mloss(pred_targ_match1, g, target_feature, g_feature, softmax) \
#                        + args.lambda_match_target2 * mloss(pred_targ_match2, g, target_feature, g_feature, softmax)
#             loss /= args.iter_size
#             loss.backward()

            S2T_nnf = bds_voting.PatchMatch(source_feature, target_feature).forward()
            T2S_nnf = bds_voting.PatchMatch(target_feature, source_feature).forward()

            prob_p = bds_voting.build_prob_map(labels.unsqueeze(0), args.num_classes)
            
            _, _, h, w = pred.size()
            _, _, fh, fw = source_feature.size()
            scale_s = torch.Tensor([h, w]) // torch.Tensor([fh, fw])
            _, _, h, w = pred_target.size()
            _, _, fh, fw = target_feature.size()
            scale_t = torch.Tensor([h, w]) // torch.Tensor([fh, fw])
            nbb_list_s, nbb_list_t = matching_loss.find_NBB(S2T_nnf, T2S_nnf, labels.unsqueeze(0), prob_p)
            cropsize_list_s = matching_loss.get_crop_size(pred, nbb_list_s, scale_s, device)
            cropsize_list_t = matching_loss.get_crop_size(pred_target, nbb_list_t, scale_t, device)

            for n in range(len(nbb_list_t)):
                pts_num = len(nbb_list_t[n])
                for k in range(len(nbb_list_t[n])):
                    ti, tj = nbb_list_t[n][k]
                    si, sj = nbb_list_s[n][k]
                    cropped_targ = matching_loss.crop_img(pred_target, scale_t, torch.Tensor([ti, tj]), cropsize_list_t[n][k])
                    cropped_source = matching_loss.crop_img(pred, scale_t, torch.Tensor([ti, tj]), cropsize_list_t[n][k])
                    cropped_label = matching_loss.crop_img(labels.unsqueeze(1), scale_s, torch.Tensor([si, sj]), cropsize_list_s[n][k])

                    loss_seg_local = seg_loss(cropped_source, cropped_label.squeeze(1))
                    loss_seg_local_value += loss_seg_local.item() / pts_num

                    D_out_t = model_D(F.softmax(cropped_targ))
                    loss_adv_local = bce_loss(D_out_t, torch.FloatTensor(D_out_t.data.size()).fill_(source_label).to(device))
                    loss_adv_local_value += loss_adv_local.item() / pts_num

                    loss_local = args.lambda_seg_local * loss_seg_local + args.lambda_adv_local * loss_adv_local
                    loss_local /= args.iter_size * pts_num
                    loss_local.backward(retain_graph=True)


            # train D

            # bring back requires_grad
            for param in model_D.parameters():
                param.requires_grad = True

            for n in range(len(nbb_list_t)):
                pt_num = len(nbb_list_t[n])
                for k in range(len(nbb_list_t[n])):
                    ti, tj = nbb_list_t[n][k]
                    si, sj = nbb_list_s[n][k]
                    cropped_targ = matching_loss.crop_img(pred_target, scale_t, torch.Tensor([ti, tj]),
                                                          cropsize_list_t[n][k])
                    cropped_source = matching_loss.crop_img(pred, scale_s, torch.Tensor([si, sj]),
                                                            cropsize_list_s[n][k])

                    D_out_t = model_D(F.softmax(cropped_targ.detach()))
                    D_out_s = model_D(F.softmax(cropped_source.detach()))

                    loss_Dt = bce_loss(D_out_t, torch.FloatTensor(D_out_t.data.size()).fill_(target_label).to(device))
                    loss_Ds = bce_loss(D_out_s, torch.FloatTensor(D_out_s.data.size()).fill_(source_label).to(device))

                    loss_D_local = loss_Dt + loss_Ds
                    loss_D_local /= pt_num * args.iter_size / 2
                    loss_D_local.backward()
                    loss_D_local_value += loss_D_local.item()

            # train with source
            pred = pred.detach()
            D_out = model_D(F.softmax(pred))
            loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
            loss_D = loss_D / args.iter_size / 2
            loss_D.backward()

            loss_D_value += loss_D.item()

            # train with target
            pred_target = pred_target.detach()
            D_out = model_D(F.softmax(pred_target))
            loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(target_label).to(device))
            loss_D = loss_D / args.iter_size / 2
            loss_D.backward()

            loss_D_value += loss_D.item()

        optimizer.step()
        optimizer_D.step()

        if args.tensorboard:
            scalar_info = {
                'loss_seg': loss_seg_value,
                'loss_seg_local': loss_seg_local_value,
                'loss_adv_target': loss_adv_target_value,
                'loss_local_adv': loss_adv_local_value,
                'loss_D': loss_D_value,
                'loss_D_local': loss_D_local_value
            }

            if i_iter % 10 == 0:
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, i_iter)

        print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {0:8d}/{1:8d}, loss_seg = {2:.3f} loss_adv = {4:.3f}, loss_D = {6:.3f} '
        'loss_seg_local = {2:.3f} loss_adv_local = {4:.3f}, loss_D_local = {6:.3f}'
            .format(i_iter, args.num_steps, loss_seg_value, loss_adv_target_value, loss_D_value, loss_seg_local_value, loss_adv_local_value, loss_D_local_value)
        )

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D.pth'))

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()
