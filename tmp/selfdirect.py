# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from loss.partial_loss import LabelWiseSignificanceCrossEntropy, AdaptiveMaskedCrossEntropyLoss
from model_define.defined_model import (KMNISTNet, CIFARNet, CIFARNet_SelfDirect, KMNISTNet_Directed,
                                        KMNISTNet_Directed_Norm, CIFARNet_SelfDirect_Norm, KMNISTNet_Directed_Dual, CIFARNet_SelfDirect_Dual)
# from model_define.hugging_face_vit import ViTForImageClassification
import torchvision.models as models
import os
import pandas as pd
torch.manual_seed(1000)
torch.cuda.manual_seed(1000)
import random
random.seed(1000)

import argparse

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--epoch', '-e', dest='epoch', default=40, help='epoch')
parser.add_argument('--dataset', '-d', dest='dataset', default="CIFAR10", help='dataset', required=False)
parser.add_argument('--opt_alg', '-a', dest='opt_alg', default="SGD", help='opt_alg', required=False)
parser.add_argument('--lossfunction', '-l', dest='lossfunction', default="MASKEDLABEL", help='lossfunction', required=False)
parser.add_argument('--model', '-m', dest='model', default="directed", help='derected|norm|base', required=False)

args = parser.parse_args()
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

########################################################################
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.


batch_size = 256

current_folder = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_folder, 'data')


def defineopt(model):
    if args.opt_alg == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5, weight_decay=0.2)
    elif args.opt_alg == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=0.5e-4)
    elif args.opt_alg == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
    else:
        raise Exception("Not accept optimizer of {}".args.opt_alg)
    return optimizer


def define_model(data, model):
    if 'mnist' in data.lower():
        if model.lower() == 'base':
            net = KMNISTNet(num_class=dataclasses_num, num_channel=num_channel)
        elif model.lower() == "directed":
            net = KMNISTNet_Directed(num_class=dataclasses_num, num_channel=num_channel)
        elif model.lower() == "norm":
            net = KMNISTNet_Directed_Norm(num_class=dataclasses_num, num_channel=num_channel)
        else:
            raise Exception("Failed to support model {}".format(args.model))
    elif "cifa" in data:
        if args.model.lower() == 'base':
            net = CIFARNet(num_class=dataclasses_num, num_channel=num_channel)
        elif args.model.lower() == "directed":
            net = CIFARNet_SelfDirect(num_class=dataclasses_num, num_channel=num_channel)
        elif args.model.lower() == "norm":
            net = CIFARNet_SelfDirect_Norm(num_class=dataclasses_num, num_channel=num_channel)
        else:
            raise Exception("Failed to support model {}".format(args.model))
    else:
        raise Exception("")

    return net

if args.dataset == "CIFAR10":
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    num_channel = 3
    image_size = trainset.data.shape[1]
    dataclasses_num = len(trainset.classes)

    net = define_model(args.dataset, args.model)
    net = net.to(device)

elif args.dataset == "CIFAR100":
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR100(root=data_path, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    num_channel = 3
    image_size = trainset.data.shape[1]
    dataclasses_num = len(trainset.classes)

    net = define_model(args.dataset, args.model)
    net = net.to(device)

elif args.dataset == 'IMAGENET':

    trainset = torchvision.datasets.ImageNet(root=data_path, split="train")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.EMNIST(root=data_path, split="mnist")
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

elif args.dataset == "EMNIST":
    transform = transforms.Compose(
        [transforms.ToTensor(), torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])

    trainset = torchvision.datasets.EMNIST(root=data_path, train=True, split="mnist",
                                             download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.EMNIST(root=data_path, train=False, split="mnist",
                                             download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    # define model
    num_channel = 1
    image_size = trainset.data.shape[1]
    dataclasses_num = len(trainset.classes)

    net = define_model(args.dataset, args.model)
    net.half()
    net = net.to(device)

elif args.dataset == "KMNIST":
    transform = transforms.Compose(
        [transforms.ToTensor(), torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])
    trainset = torchvision.datasets.KMNIST(root=data_path, train=True,
                                           download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.KMNIST(root=data_path, train=False,
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    # define model
    num_channel = 1
    image_size = trainset.data.shape[1]
    dataclasses_num = len(trainset.classes)

    net = define_model(args.dataset, args.model)
    net = net.to(device)

elif args.dataset == "FashionMNIST":
    transform = transforms.Compose(
        [transforms.ToTensor(), torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])
    trainset = torchvision.datasets.FashionMNIST(root=data_path, train=True,
                                           download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.FashionMNIST(root=data_path, train=False,
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    # define model
    num_channel = 1
    image_size = trainset.data.shape[1]
    dataclasses_num = len(trainset.classes)

    net = define_model(args.dataset, args.model)
    net = net.to(device)

elif args.dataset == "QMNIST":
    transform = transforms.Compose(
        [transforms.ToTensor(), torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])
    trainset = torchvision.datasets.QMNIST(root=data_path, train=True,
                                           download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.QMNIST(root=data_path, train=False,
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    # define model
    num_channel = 1
    image_size = trainset.data.shape[1]
    dataclasses_num = len(trainset.classes)
    net = define_model(args.dataset, args.model)
    net = net.to(device)

else:
    raise Exception("Unsupport dataset type {}".format(args.dataset))


criterion = nn.CrossEntropyLoss()

optimizer = defineopt(net)
########################################################################
def run_test(model_path):
    correct = 0
    total = 0
    net.load_state_dict(torch.load(model_path))
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            images = images.to(device)
            if isinstance(net, KMNISTNet_Directed) or isinstance(net, CIFARNet_SelfDirect):
                outputs, loss_directed = net(images)
            elif isinstance(net, KMNISTNet_Directed_Dual) or isinstance(net, CIFARNet_SelfDirect_Dual):
                outputs, _, _ = net(images)
            elif isinstance(net, KMNISTNet_Directed_Norm):
                outputs = net(images)
            elif isinstance(net, KMNISTNet) or isinstance(net, CIFARNet):
                outputs = net(images)
            else:
                raise Exception("Unsupport model type")

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            labels = labels.to(device)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def L2_reg(parameters):
    L2 = 0
    for p in parameters:
        L2 += torch.sum(torch.square(p))
    return torch.round(L2).item()

def save_model(net, model_path):
    torch.save(net.state_dict(), model_path)

def reinitialization_model(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
# 4. Train the network
for t in range(10): # train model 10 times
    acc = []
    for epoch in range(int(args.epoch)):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # forward + backward + optimize
            inputs = inputs.to(device)
            labels = labels.to(device)
            if isinstance(net, KMNISTNet_Directed) or isinstance(net, CIFARNet_SelfDirect):
                outputs, loss_directed = net(inputs)
                loss = criterion(outputs, labels)

                for l in loss_directed:
                    loss += torch.sum(torch.square(l))
            elif isinstance(net, KMNISTNet_Directed_Dual) or isinstance(net, CIFARNet_SelfDirect_Dual):
                outputs, loss_plan, loss_channel = net(inputs)
                loss = criterion(outputs, labels)

                # max of each channel signal
                for l in loss_plan:
                    loss -= torch.sum(torch.square(l))
                # minimum of channels signal
                for l in loss_channel:
                    loss += torch.sum(torch.square(l))

            elif isinstance(net, KMNISTNet_Directed_Norm) or isinstance(net, CIFARNet_SelfDirect_Norm):
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            elif isinstance(net, KMNISTNet) or isinstance(net, CIFARNet):
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            # zero the parameter gradients
            optimizer.zero_grad()

            running_loss += loss.item()
            # print("{} step loss is {}".format(i, loss.item()))
        model_path = os.path.join(current_folder, '../model', '{}_{}_{}_net.pth'.format(args.dataset, args.opt_alg, args.lossfunction))
        save_model(net, model_path)
        acc_epoch = run_test(model_path)
        acc_epoch = round(acc_epoch, 2)
        L2 = L2_reg(net.parameters())
        acc.append([epoch, acc_epoch, round(running_loss, 2), L2])
        print("{} epoch acc is {}, L2 is {}".format(epoch, acc_epoch, L2))
    print('Finished Training')
    result_file = os.path.join(os.path.join(current_folder, 'result',
                                            'result_{}_{}_{}_{}'.format(args.model, args.dataset,
                                                                               args.opt_alg, args.lossfunction), "{}.csv".format(str(t))))
    if not os.path.exists(os.path.dirname(result_file)):
        os.makedirs(os.path.dirname(result_file))
    pd.DataFrame(acc).to_csv(result_file, header=["epoch", "training_acc", "training_loss", "L2"], index=False)
    del net
    del optimizer
    net = define_model(args.dataset, args.model)
    net.half()
    net = net.to(device)
    optimizer = defineopt(net)
