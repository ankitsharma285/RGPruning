import argparse
import numpy as np
import os,sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from compute_flops import print_model_param_nums, print_model_param_flops

from models import *


# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--data', type=str, default=None,
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=19,
                    help='depth of the vgg')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--arch', type=str, default='vgg',
                    help='vgg')
parser.add_argument('--multi_GPU', type=int, default=0,
                    help='multi_GPU')
# multi-gpus
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

gpu_ids = args.gpu_ids.split(',')
args.gpu_ids = []
for gpu_id in gpu_ids:
   id = int(gpu_id)
   if id > 0:
       args.gpu_ids.append(id)
if len(args.gpu_ids) > 0:
   torch.cuda.set_device(args.gpu_ids[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == "imagenet":
    model = slimmingvgg(dataset=args.dataset, depth=args.depth)
else:
    model = vgg(dataset=args.dataset, depth=args.depth)

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        # args.start_epoch = checkpoint['epoch']
        # best_prec1 = checkpoint['best_prec1']
        if args.multi_GPU:
            try:
                print('hello')
                model.load_state_dict(checkpoint['state_dict'])
            except:
                model = torch.nn.DataParallel(model, device_ids=args.gpu_ids).cuda()
                model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        # print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              # .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.model))
        exit()

print('original model param: ', print_model_param_nums(model))
if args.dataset == 'imagenet':
    # print('original model flops: ', print_model_param_flops(model, 224, True))
    pass
else:
    print('original model flops: ', print_model_param_flops(model, 32, True))

if args.cuda:
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

print('layer_count: {:d}'.format(layer_count))
print(layer_thre)

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
        cfg.append('M')

print('Pre-processing Successful!')

# apply mask flops
# p_flops += total

# compare two mask distance
print('Pre-processing Successful!')

# simple test model after Pre-processing prune (simple set BN scales to zeros)
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Make real prune
print(cfg)
if args.dataset == "imagenet":
    newmodel = slimmingvgg(dataset=args.dataset, cfg=cfg)
else:
    newmodel = vgg(dataset=args.dataset, cfg=cfg)

if len(args.gpu_ids) > 1:
    newmodel = torch.nn.DataParallel(newmodel, device_ids=args.gpu_ids)
if args.cuda:
    newmodel.cuda()

num_parameters = sum([param.nelement() for param in newmodel.parameters()])
savepath = os.path.join(args.save, "prune.txt")
with open(savepath, "w") as fp:
    fp.write("Configuration: \n"+str(cfg)+"\n")
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    # fp.write("Test accuracy: \n"+str(acc))

layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.BatchNorm2d):
        if torch.sum(end_mask) == 0:
            continue
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        if torch.sum(end_mask) == 0:
            continue
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        # random set for test
        # new_end_mask = np.asarray(end_mask.cpu().numpy())
        # new_end_mask = np.append(new_end_mask[int(len(new_end_mask)/2):], new_end_mask[:int(len(new_end_mask)/2)])
        # idx1 = np.squeeze(np.argwhere(new_end_mask))

        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
        w1 = w1[idx1.tolist(), :, :, :].clone()
        m1.weight.data = w1.clone()
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))

# print(newmodel)
model = newmodel
# test(model)
param = print_model_param_nums(model)
flops = print_model_param_flops(model.cpu(), 32, True)
with open(savepath, "w") as fp:
    fp.write("new model param: \n"+str(param)+"\n")
    fp.write("new model flops: \n"+str(flops)+"\n")
print('new model param: ', param)
print('new model flops: ', flops)
