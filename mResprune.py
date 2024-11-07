import argparse
import numpy as np
import os, sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from compute_flops import print_model_param_nums, print_model_param_flops

import models
from models import *
# from models.preresnet_imagenet import BasicBlock


# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar10)')
parser.add_argument('--data', type=str, default=None,
                    help='path to dataset')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=164,
                    help='depth of the resnet')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--init', default='', type=str, metavar='PATH',
                    help='path to the rewind init (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--arch', default="resnet", type=str, help='model name')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(1)
if args.cuda:
    torch.cuda.manual_seed(1)

if not os.path.exists(args.save):
    os.makedirs(args.save)

if args.dataset == 'tiny':
    model = models.__dict__[args.arch](pretrained=False,num_classes=200)
elif args.dataset == "celeba":
    model = models.__dict__[args.arch](pretrained=False,num_classes=40)
    # model = PreResNet(BasicBlock, [2, 2, 2, 2])
else:
    model = resnet(depth=args.depth, dataset=args.dataset)

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        exit()

if args.dataset == 'tiny':
    print('original model param: ', print_model_param_nums(model))
    print('original model flops: ', print_model_param_flops(model, 64, True))
elif args.dataset == 'celeba':
    print('original model param: ', print_model_param_nums(model))
    print('original model flops: ', print_model_param_flops(model, 128, True))
else:
    print('original model param: ', print_model_param_nums(model))
    print('original model flops: ', print_model_param_flops(model, 32, True))

model.cuda()

layer_count = 0
layer_thre = []
for m in model.modules():
    index = 0
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn = torch.zeros(size)
        bn[0:size] = m.weight.data.abs().clone()
        y, i = torch.sort(bn)
        thre_index = int(size * args.percent)
        layer_thre.append(y[thre_index])
        layer_count += 1

cfg = []
cfg_mask = []
layer_count = 0
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        pruned = 0
        weight_copy = m.weight.data.abs().clone()
        thre = layer_thre[layer_count]
        mask = weight_copy.gt(thre.cuda()).float().cuda()
        pruned = mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
        layer_count += 1
        pruned_ratio = pruned/mask.shape[0]
    elif isinstance(m, nn.MaxPool2d):
        # cfg.append('M')
        pass


print("Cfg:")
print(cfg)


if args.dataset == 'tiny':
    downsample = [7, 12, 17]
    cfg = [item for i, item in enumerate(cfg) if i not in downsample]
    cfg_mask = [item for i, item in enumerate(cfg_mask) if i not in downsample]
    newmodel = models.__dict__[args.arch](pretrained=False, cfg=cfg, num_classes=200)
    # newmodel = PreResNet(BasicBlock, [2, 2, 2, 2], cfg=cfg)
elif args.dataset == 'celeba':
    downsample = [7, 12, 17]
    cfg = [item for i, item in enumerate(cfg) if i not in downsample]
    cfg_mask = [item for i, item in enumerate(cfg_mask) if i not in downsample]
    newmodel = models.__dict__[args.arch](pretrained=False, cfg=cfg, num_classes=40)
else:
    newmodel = resnet(depth=args.depth, dataset=args.dataset, cfg=cfg)

if args.init != '':
    if os.path.isfile(args.init):
        print("=> loading checkpoint '{}'".format(args.init))
        checkpoint = torch.load(args.init)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.init))

newmodel.cuda()
#     model.cuda()

num_parameters = sum([param.nelement() for param in newmodel.parameters()])
savepath = os.path.join(args.save, "prune.txt")
with open(savepath, "w") as fp:
    fp.write("Configuration: \n"+str(cfg)+"\n")
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    # fp.write("Test accuracy: \n"+str(acc))

old_modules = list(model.modules())
new_modules = list(newmodel.modules())

useful_i = []
for i, module in enumerate(old_modules):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Linear) or isinstance(module, nn.ReLU) or isinstance(module, channel_selection):
        useful_i.append(i)
temp = []
for i, item in enumerate(useful_i):
    temp.append(old_modules[item])
# for i, item in enumerate(temp):
#     print(i, item)
# sys.exit()

layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
conv_count = 0
bn_count = 0

if args.dataset == 'celeba':
    downsample = [8, 13, 18]
    last_block = [3, 5, 7, 10, 12, 15, 17, 20]
    for layer_id in range(len(temp)):
        m0 = old_modules[useful_i[layer_id]]
        m1 = new_modules[useful_i[layer_id]]
        # print(m0)
        if isinstance(m0, nn.BatchNorm2d):
            bn_count += 1
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print(layer_id_in_cfg, len(cfg_mask))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))

            if bn_count == 1:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                m2 = new_modules[useful_i[layer_id+2]] # channel selection
                assert isinstance(m2, channel_selection)
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]

            elif bn_count in downsample:
                # If the current layer is the downsample layer, then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

            elif bn_count in last_block:
                # If the current layer is the last conv-bn layer in block, then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                # We need to set the channel selection layer.
                if bn_count + 1 in downsample:
                    m2 = new_modules[useful_i[layer_id+3]]
                    assert isinstance(m2, channel_selection)
                else:
                    m2 = new_modules[useful_i[layer_id+1]]
                    assert isinstance(m2, channel_selection) or isinstance(m2, nn.Linear)
                if isinstance(m2, channel_selection):
                    m2.indexes.data.zero_()
                    m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            else:
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            conv_count += 1
            if conv_count == 1:
                m1.weight.data = m0.weight.data.clone()
                continue
            elif conv_count in downsample: # downsample
                # We need to consider the case where there are downsampling convolutions.
                # For these convolutions, we just copy the weights.
                m1.weight.data = m0.weight.data.clone()
                continue
            elif conv_count in last_block:
                # the last convolution in the residual block.
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                m1.weight.data = w1.clone()
                continue
            else:
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            # end_mask = cfg_mask[-1]
            # idx0 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print(idx0)
            # if idx0.size == 1:
            #     idx0 = np.resize(idx0, (1,))
            # m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
else:
    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]
        if isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))

            if isinstance(old_modules[layer_id + 1], channel_selection):
                # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                # We need to set the channel selection layer.
                m2 = new_modules[layer_id + 1]
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            else:
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            if conv_count == 0:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
            if isinstance(old_modules[layer_id-1], channel_selection) or isinstance(old_modules[layer_id-1], nn.BatchNorm2d):
                # This convers the convolutions in the residual block.
                # The convolutions are either after the channel selection layer or after the batch normalization layer.
                conv_count += 1
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()

                # If the current convolution is not the last convolution in the residual block, then we can change the
                # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
                if conv_count % 3 != 1:
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
                continue

            # We need to consider the case where there are downsampling convolutions.
            # For these convolutions, we just copy the weights.
            m1.weight.data = m0.weight.data.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))

# print(newmodel)
model = newmodel
param = print_model_param_nums(model)
if args.dataset == 'tiny':
    flops = print_model_param_flops(model.cpu(), 64, True)
elif args.dataset == 'celeba':
    flops = print_model_param_flops(model.cpu(), 128, True)
else:
    flops = print_model_param_flops(model.cpu(), 32, True)
print('new model param: ', param)
print('new model flops: ', flops)
