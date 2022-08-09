# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms,models
from torch.autograd import Variable
from LWENet import lwenet

from torch.backends import cudnn
import numpy as np
from time import *
import utils
import os
from torchsummary import summary
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=0.0005, metavar='wd',
                    help='weight_decay (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')

#cuda related
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
else:
    args.gpu = None

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}



valid_path = "E:\陈梦飞\\New_work_2\dataset\BOSSBase_256\WOW\\valid_boss\WOW-0.4\\"
test_path = "E:\陈梦飞\\New_work_2\dataset\BOSSBase_256\WOW\\test_boss\WOW-0.4\\"
train_cover_path = "E:\陈梦飞\\New_work_2\dataset\BOSSBase_256\WOW\\train_boss_BOSW2\WOW-0.4\\0\\"
train_stego_path = "E:\陈梦飞\\New_work_2\dataset\BOSSBase_256\WOW\\train_boss_BOSW2\WOW-0.4\\1\\"

# valid_path = "E:\陈梦飞\\New_work_2\dataset\BOSSBase_256\HILL\\valid_boss\HILL-0.2\\"
# test_path = "E:\陈梦飞\\New_work_2\dataset\BOSSBase_256\HILL\\test_boss\HILL-0.2\\"
# train_cover_path = "E:\陈梦飞\\New_work_2\dataset\BOSSBase_256\HILL\\train_boss_BOSW2\HILL-0.2\\0\\"
# train_stego_path = "E:\陈梦飞\\New_work_2\dataset\BOSSBase_256\HILL\\train_boss_BOSW2\HILL-0.2\\1\\"

# valid_path = "E:\陈梦飞\\New_work_2\dataset\BOSSBase_256\S_UNIWARD\\valid_boss\SUN-0.4\\"
# test_path = "E:\陈梦飞\\New_work_2\dataset\BOSSBase_256\S_UNIWARD\\test_boss\SUN-0.4\\"
# train_cover_path = "E:\陈梦飞\\New_work_2\dataset\BOSSBase_256\S_UNIWARD\\train_boss_BOSW2\SUN-0.4\\0\\"
# train_stego_path = "E:\陈梦飞\\New_work_2\dataset\BOSSBase_256\S_UNIWARD\\train_boss_BOSW2\SUN-0.4\\1\\"


print('torch ',torch.__version__)
print('train_path = ',train_cover_path)
print('valid_path = ',valid_path)
print('test1_path = ',test_path)
print('train_batch_size = ',args.batch_size)
print('test_batch_size = ',args.test_batch_size)
train_transform = transforms.Compose([utils.AugData(),utils.ToTensor()])#成对训练并使用虚拟增强（VA）时顺序必须为utils.AugData(),utils.ToTensor()，若只使用成对训练则utils.ToTensor()

train_data= utils.DatasetPair(train_cover_path,train_stego_path,train_transform)
valid_data= datasets.ImageFolder(valid_path, transform=transforms.Compose([transforms.Grayscale(),transforms.ToTensor()]))
test1_data= datasets.ImageFolder(test_path,
            transform=transforms.Compose([transforms.Grayscale(),transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(train_data,batch_size=args.batch_size, shuffle=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(valid_data,batch_size=args.test_batch_size, shuffle=False, **kwargs)
test1_loader = torch.utils.data.DataLoader(test1_data,batch_size=args.test_batch_size, shuffle=True, **kwargs)


model = lwenet()
print(model)
print(summary(model.cuda(),(1,256,256)))
if args.cuda:
    model.cuda()
cudnn.benchmark = True


def initWeights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')
model.apply(initWeights)

params = model.parameters()
params_wd, params_rest = [], []
for param_item in params:
    if param_item.requires_grad:
        (params_wd if param_item.dim() != 1 else params_rest).append(param_item)

param_groups= [{'params': params_wd, 'weight_decay': args.weight_decay},
                    {'params': params_rest}]



optimizer = optim.SGD(param_groups, lr=args.lr,momentum=args.momentum)

DECAY_EPOCH = [80,140,180]
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=DECAY_EPOCH,gamma=0.1)


def train(epoch):
    total_loss =0
    lr_train = (optimizer.state_dict()['param_groups'][0]['lr'])
    print(lr_train)

    model.train()
    for batch_idx, data in enumerate(train_loader):
        if args.cuda:
            data, label = data['images'].cuda(), data['labels'].cuda()
        data, label = Variable(data), Variable(label)


        if batch_idx == len(train_loader) - 1:
            last_batch_size = len(os.listdir(train_cover_path)) - args.batch_size * (len(train_loader) - 1)
            datas = data.view(last_batch_size * 2, 1, 256, 256)
            labels = label.view(last_batch_size * 2)
        else:
            datas = data.view(args.batch_size * 2, 1, 256, 256)
            labels = label.view(args.batch_size * 2)
        optimizer.zero_grad()
        output = model(datas)
        output1 = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output1, labels)
        total_loss = total_loss+loss.item()
        loss.backward()
        optimizer.step()
        
        if (batch_idx+1) % args.log_interval == 0:
            b_pred = output.max(1, keepdim=True)[1]
            b_correct = b_pred.eq(labels.view_as(b_pred)).sum().item()

            b_accu=b_correct/(labels.size(0))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\ttrain_accuracy: {:.6f}\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                100. * (batch_idx+1) / len(train_loader),b_accu ,loss.item()))
    print('train Epoch: {}\tavgLoss: {:.6f}'.format(epoch,total_loss/len(train_loader)))
    scheduler.step()
    


def test():
    model.eval()
    test1_loss = 0
    correct = 0.
    with torch.no_grad():
        for data, target in test1_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test1_loss += F.nll_loss(F.log_softmax(output, dim=1), target,reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test1_loss /= len(test1_loader.dataset)
    print('Test1 set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
    test1_loss, correct, len(test1_loader.dataset),
    100. * correct / len(test1_loader.dataset)))
    accu=float(correct) / len(test1_loader.dataset)
    return accu,test1_loss
    


def valid():
    model.eval()
    valid_loss = 0
    correct = 0.
    with torch.no_grad():
        for data, target in valid_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            valid_loss += F.nll_loss(F.log_softmax(output, dim=1), target,reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    valid_loss /= len(valid_loader.dataset)
    print('valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)'.format(
    valid_loss, correct, len(valid_loader.dataset),
    100. * correct / len(valid_loader.dataset)))
    accu=float(correct) / len(valid_loader.dataset)
    return accu,valid_loss



def sum(pred,target):
    #此代码不包含统计所有的载体图片和所有载密图像的数量，需要在调用后设置所有载体图像的数量。
    #print(len(target))
    pred = pred.view_as(target)
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    l1 = []
    for i in range(len(target)):
        l1.append(pred[i]+target[i])
    #print(l1.count(0))
    #print(l1.count(2))
    # l1.count(0)即为 正确被判定为载体图像（阴性）的数量。l1.count(2)，即为正确被判定为载密图像（阳性）的数量。l1.count(0)+l1.count(2) 即为判断正确的总个数
    return l1.count(0),l1.count(2),l1.count(0)+l1.count(2)

def valid_mulit():
    model.eval()
    valid_loss = 0
    correct = 0.
    accu = 0.
    N = 0  #正确被分类为载体图像的数目
    P = 0 # 正确被分类为载密图像的数目

    with torch.no_grad():
        for data, target in valid_loader:

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            output = F.log_softmax(output,dim=1)
            valid_loss += F.nll_loss(output, target,reduction='sum').item()

            pred = output.max(1, keepdim=True)[1]
            a, b, c = sum(pred, target)
            N += a
            P += b
            correct += c
    valid_loss /= len(valid_loader.dataset)
    accu=float(correct) / len(valid_loader.dataset)
    print('Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)'.format(
    valid_loss, correct, len(valid_loader.dataset),
    100. * accu))
    S = len(valid_loader.dataset)/2 # 待测数据中所有的载密图像个数  具体的数量具体设置，如果载体图像等于载密数量则这样写代码即可
    C=  len(valid_loader.dataset)/2 # 待测数据集中所有载体图像的个数
    FPR = (C-N)/C # 虚警率 即代表载体图像被误判成载密图像 占所有载体图像的比率
    Pmd = (S-P)/S # 漏检率 即代表载密图像被误判成载体图像 占所有载密图像的比率
    print('Valid set 虚警率(FPR): {}/{} ({:.6f}%)'.format(C - N , C , 100. * FPR))
    print('Valid set 漏检率(FNR): {}/{} ({:.6f}%)'.format(S - P , S , 100. * Pmd)) #名称定义来自于  来自于软件学报 论文 《基于深度学习的图像隐写分析综述》Journal of Software,2021,32(2):551−578 [doi: 10.13328/j.cnki.jos.006135]
    return accu,valid_loss



t1 = time()

for epoch in range(1, args.epochs+1):#

    train(epoch)
    #torch.save(model.state_dict(), '.\\wow-0.4_boss_bosw2_VA\\'+str(epoch)+'.pkl',_use_new_zipfile_serialization = False)#保存每个epoc之后的网络参数
    valid()
    test()



t2 = time()
print(t2 - t1)

# t1 = time()
# for i in range(161,201):
#     a = 'E:\陈梦飞\\New_work_2\\Rep_convnet\\7\\ablation\\net_params\DenseNet_Add_GAP_mutil_feat_last_DWSConv\\wow-0.4_boss_bosw2\\'+str(i)+'.pkl'
#     model.load_state_dict(torch.load(a))
#     print(a)
#     t3 = time()
#     valid()
#     t4 = time()
#     print('valid time = ',t4-t3)
#     t5 = time()
#     test()
#     t6 =  time()
#     print('test time = ', t6 - t5)
# t2 = time()
# print('total_test_time = ',t2 -t1)




