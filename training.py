# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from loss.partial_loss import LabelWiseSignificanceCrossEntropy, AdaptiveMaskedCrossEntropyLoss
from model_define.defined_model import KMNISTNet, CIFARNet, CIFARNet_Infer, IMAGENET
from model_define.resnet import ResNet18, ResNet50
from cv.define_model import ViTForImageClassification
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models
import os
import pandas as pd
torch.manual_seed(1000)
torch.cuda.manual_seed(1000)
import random
random.seed(1000)
import argparse

TINY_IMAGENET_DATASET = 'Maysee/tiny-imagenet'
# IMAGENET1k_DATASET = 'imagenet-1k'
POKEMON_DATASET = "keremberke/pokemon-classification"
OXFORD_FLOWER_DATASET = "nelorth/oxford-flowers"

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--epoch', '-e', dest='epoch', default=100, help='epoch')
parser.add_argument('--dataset', '-d', dest='dataset', default="CIFAR10", help='dataset', required=False)
parser.add_argument('--opt_alg', '-a', dest='opt_alg', default="SGD", help='opt_alg', required=False)
parser.add_argument('--lossfunction', '-l', dest='lossfunction', default="MASKEDLABEL", help='lossfunction', required=False)
parser.add_argument('--lr', '-lr', dest='lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--runs', dest='runs', type=int, default=10, help='number of independent training runs')
parser.add_argument('--alpha', dest='alpha', type=float, default=0.2, help='alpha of LWSCE / ALWSCE')
parser.add_argument('--keep_hardest', dest='keep_hardest', action=argparse.BooleanOptionalAction, default=True,
                    help='LWSCE never masks the most probable non-target class; --no-keep_hardest is the masking of pilot 2')
parser.add_argument('--tag', dest='tag', default='', help='suffix for result and model names, e.g. lr1e-3')
parser.add_argument('--model', dest='model', default='default', choices=['default', 'resnet18', 'resnet50'],
                    help='network; default is the dataset specific one, resnets support CIFAR10 / CIFAR100 only')
parser.add_argument('--augment', dest='augment', action='store_true',
                    help='random crop and horizontal flip on the CIFAR training set')
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.0)
parser.add_argument('--scheduler', dest='scheduler', default='none', choices=['none', 'cosine'],
                    help='cosine anneals the lr to 0 over the epochs of a run')
parser.add_argument('--amp', dest='amp', action='store_true', help='mixed precision training on GPU')
parser.add_argument('--num_workers', dest='num_workers', type=int, default=0)
parser.add_argument('--seed_mode', dest='seed_mode', default='per_run', choices=['per_run', 'global'],
                    help='per_run reseeds every run with SEED + run, so a single run can be reproduced on its own; '
                         'global seeds once, as the first pilot did')


args = parser.parse_args()
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")
use_amp = args.amp and device.type == 'cuda'
SEED = 1000

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

########################################################################
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.


batch_size = 128

current_folder = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_folder, 'data')

VAL_RATIO = 0.1
SPLIT_SEED = 1000

def split_train_val(trainset, evalset=None):
    # hold out a fixed part of the training set for model selection,
    # so every run and loss function sees the same split and the test set stays unseen.
    # evalset is the same data without augmentation, used for the validation part
    val_size = int(len(trainset) * VAL_RATIO)
    train_subset, val_subset = random_split(trainset, [len(trainset) - val_size, val_size],
                                            generator=torch.Generator().manual_seed(SPLIT_SEED))
    if evalset is not None:
        val_subset = torch.utils.data.Subset(evalset, val_subset.indices)
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size,
                                              shuffle=True, num_workers=args.num_workers)
    valloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size,
                                            shuffle=False, num_workers=args.num_workers)
    return trainloader, valloader

def cifar_train_transform(base):
    if not args.augment:
        return base
    return transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
                              + base.transforms)

if args.augment and args.dataset not in ("CIFAR10", "CIFAR100"):
    raise Exception("--augment supports CIFAR10 / CIFAR100 only")

if args.dataset == "CIFAR10":

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            download=True, transform=cifar_train_transform(transform))
    evalset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                           download=True, transform=transform)
    trainloader, valloader = split_train_val(trainset, evalset)

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=args.num_workers)
    num_channel = 3
    image_size = trainset.data.shape[1]
    dataclasses_num = len(trainset.classes)

    make_net = lambda: CIFARNet(num_class=dataclasses_num, num_channel=num_channel)


elif args.dataset == "CIFAR100":
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                            download=True, transform=cifar_train_transform(transform))
    evalset = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                           download=True, transform=transform)
    trainloader, valloader = split_train_val(trainset, evalset)

    testset = torchvision.datasets.CIFAR100(root=data_path, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=args.num_workers)
    num_channel = 3
    image_size = trainset.data.shape[1]
    dataclasses_num = len(trainset.classes)

    # net = ViTForImageClassification(num_labels=dataclasses_num)
    make_net = lambda: CIFARNet(num_class=dataclasses_num, num_channel=num_channel)

elif args.dataset == 'IMAGENET':
    from cv.load_data import load_imagenet_dataset
    trainloader, valloader, testloader, dataclasses_num, image_size = load_imagenet_dataset(TINY_IMAGENET_DATASET)
    # net = ViTForImageClassification(num_labels=dataclasses_num, image_size=image_size)
    make_net = lambda: IMAGENET(num_class=dataclasses_num, num_channel=3)

elif args.dataset == 'POKEMON':
    from cv.load_data import load_pokemon_dataset
    trainloader, valloader, testloader, dataclasses_num, image_size = load_pokemon_dataset(POKEMON_DATASET)
    make_net = lambda: ViTForImageClassification(num_labels=dataclasses_num, image_size=image_size)
    # net = IMAGENET(num_class=dataclasses_num, num_channel=3)

elif args.dataset == 'OXFORDFLOWER':
    from cv.load_data import load_imagenet_dataset
    trainloader, valloader, testloader, dataclasses_num, image_size = load_imagenet_dataset(OXFORD_FLOWER_DATASET)
    # config = ViTConfig(image_size=64, num_labels=dataclasses_num)
    make_net = lambda: ViTForImageClassification(num_labels=dataclasses_num, image_size=image_size)
    # net = IMAGENET(num_class=dataclasses_num, num_channel=3)

elif args.dataset == "EMNIST":
    transform = transforms.Compose(
        [transforms.ToTensor(), torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])

    trainset = torchvision.datasets.EMNIST(root=data_path, train=True, split="mnist",
                                             download=True, transform=transform)
    trainloader, valloader = split_train_val(trainset)

    testset = torchvision.datasets.EMNIST(root=data_path, train=False, split="mnist",
                                             download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    # define model
    num_channel = 1
    image_size = trainset.data.shape[1]
    dataclasses_num = len(trainset.classes)

    make_net = lambda: KMNISTNet(num_class=dataclasses_num, num_channel=num_channel)

elif args.dataset == "FashionMNIST":
    transform = transforms.Compose(
        [transforms.ToTensor(), torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])
    trainset = torchvision.datasets.FashionMNIST(root=data_path, train=True,
                                           download=True, transform=transform)
    trainloader, valloader = split_train_val(trainset)

    testset = torchvision.datasets.FashionMNIST(root=data_path, train=False,
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    # define model
    num_channel = 1
    image_size = trainset.data.shape[1]
    dataclasses_num = len(trainset.classes)

    make_net = lambda: KMNISTNet(num_class=dataclasses_num, num_channel=num_channel)

elif args.dataset == "MNIST":
    transform = transforms.Compose(
        [transforms.ToTensor(), torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root=data_path, train=True,
                                           download=True, transform=transform)
    trainloader, valloader = split_train_val(trainset)

    testset = torchvision.datasets.MNIST(root=data_path, train=False,
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    # define model
    num_channel = 1
    image_size = trainset.data.shape[1]
    dataclasses_num = len(trainset.classes)

    make_net = lambda: KMNISTNet(num_class=dataclasses_num, num_channel=num_channel)


elif args.dataset == "KMNIST":
    transform = transforms.Compose(
        [transforms.ToTensor(), torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])
    trainset = torchvision.datasets.KMNIST(root=data_path, train=True,
                                           download=True, transform=transform)
    trainloader, valloader = split_train_val(trainset)

    testset = torchvision.datasets.KMNIST(root=data_path, train=False,
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    # define model
    num_channel = 1
    image_size = trainset.data.shape[1]
    dataclasses_num = len(trainset.classes)

    make_net = lambda: KMNISTNet(num_class=dataclasses_num, num_channel=num_channel)

elif args.dataset == "QMNIST":
    transform = transforms.Compose(
        [transforms.ToTensor(), torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])
    trainset = torchvision.datasets.QMNIST(root=data_path, train=True,
                                           download=True, transform=transform)
    trainloader, valloader = split_train_val(trainset)

    testset = torchvision.datasets.QMNIST(root=data_path, train=False,
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    # define model
    num_channel = 1
    image_size = trainset.data.shape[1]
    dataclasses_num = len(trainset.classes)
    make_net = lambda: KMNISTNet(num_class=dataclasses_num, num_channel=num_channel)
else:
    raise Exception("Unable to support the data {}".format(args.dataset))

if args.model != 'default':
    if args.dataset not in ("CIFAR10", "CIFAR100"):
        raise Exception("--model {} supports CIFAR10 / CIFAR100 only".format(args.model))
    resnet = {'resnet18': ResNet18, 'resnet50': ResNet50}[args.model]
    make_net = lambda: resnet(num_class=dataclasses_num, num_channel=num_channel)

if args.lossfunction == "LWSCE":
    criterion = LabelWiseSignificanceCrossEntropy(alpha=args.alpha, num_class=dataclasses_num, keep_hardest=args.keep_hardest, device=device)
elif args.lossfunction == 'CROSSENTROPY':
    criterion = nn.CrossEntropyLoss()
elif args.lossfunction == 'LABELSMOOTHING':
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
elif args.lossfunction == "ALWSCE":
    criterion = AdaptiveMaskedCrossEntropyLoss(alpha=args.alpha, num_class=dataclasses_num, device=device)
elif args.lossfunction == "LWSCE+LS":
    criterion = LabelWiseSignificanceCrossEntropy(alpha=args.alpha, num_class=dataclasses_num, keep_hardest=args.keep_hardest, label_smoothing=0.2, device=device)
else:
    raise Exception("Unaccept loss function {}".format(args.lossfunction))


########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).


# from model_define.hugging_face_vit import ViTForImageClassification

def defineopt(model):
    kwargs = {'lr': args.lr, 'weight_decay': args.weight_decay}
    if args.opt_alg == 'SGD':
        optimizer = optim.SGD(model.parameters(), **kwargs)
    elif args.opt_alg == "ADAM":
        optimizer = optim.Adam(model.parameters(), **kwargs)
    elif args.opt_alg == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), **kwargs)
    elif args.opt_alg == "LWADAM":
        from opt.customeropt import LWADAM
        optimizer = LWADAM(model.parameters(), **kwargs)
    elif args.opt_alg == "RADAM":
        optimizer = optim.RAdam(model.parameters(), **kwargs)
    elif args.opt_alg == "ADADELTA":
        optimizer = optim.Adadelta(model.parameters(), **kwargs)
    elif args.opt_alg == "ADAMW":
        optimizer = optim.AdamW(model.parameters(), **kwargs)
    else:
        raise Exception("Not accept optimizer of {}".format(args.opt_alg))
    return optimizer
def define_scheduler(optimizer):
    if args.scheduler == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=int(args.epoch) - 1)
    return None
########################################################################
def run_test(loader):
    correct = 0
    total = 0
    net.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in loader:
            # calculate outputs by running images through the network
            if "IMAGENET" in args.dataset:
                images = data["image"].to(device)
                labels = data["label"].to(device)
            elif args.dataset == 'POKEMON':
                images = data["image"].to(device)
                labels = data["labels"].to(device)
            else:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            predicted = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            labels = labels.to(device)
            correct += (predicted == labels).sum().item()

    net.train()
    return 100 * correct / total

def L2_reg(parameters):
    L2 = 0
    for p in parameters:
        L2 += torch.sum(torch.square(p))
    return torch.round(L2).item()

def save_model(net, model_path):
    torch.save(net.state_dict(), model_path)


# 4. Train the network
run_name = '{}_{}_{}'.format(args.dataset, args.opt_alg, args.lossfunction)
if args.model != 'default':
    run_name = '{}_{}'.format(args.model, run_name)
if args.tag:
    run_name = '{}_{}'.format(run_name, args.tag)
result_dir = os.path.join(current_folder, 'result', '{}_result'.format(run_name))
os.makedirs(result_dir, exist_ok=True)
os.makedirs(os.path.join(current_folder, 'model'), exist_ok=True)
model_path = os.path.join(current_folder, 'model', '{}_net.pth'.format(run_name))
# a non-target class counts as low probability when LWSCE would mask it: prob <= 2 * alpha / num_class
low_prob_threshold = 2 * args.alpha / dataclasses_num
summary = []
for t in range(args.runs): # train model args.runs times
    if args.seed_mode == 'per_run':
        torch.manual_seed(SEED + t)
        random.seed(SEED + t)
    net = make_net().to(device)
    optimizer = defineopt(net)
    scheduler = define_scheduler(optimizer)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    acc = []
    best_val_acc, best_epoch, test_acc_at_best = -1, -1, -1
    for epoch in range(1,int(args.epoch)):  # loop over the dataset multiple times
        running_loss = 0.0
        # diagnostics, summed on the device and read once per epoch
        train_correct = torch.zeros((), device=device)
        low_prob_sum = torch.zeros((), device=device)
        logit_std_sum = torch.zeros((), device=device)
        seen, batches = 0, 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]

            # forward + backward + optimize
            if "IMAGENET" in args.dataset:
                inputs = data["image"].to(device)
                labels = data["label"].to(device)
            elif args.dataset == 'POKEMON':
                inputs = data["image"].to(device)
                labels = data["labels"].to(device)
            else:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = net(inputs)
            # the loss is computed in float32, masking writes -inf into the logits
            outputs = outputs.float()

            loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # zero the parameter gradients
            optimizer.zero_grad()

            running_loss += loss.item()
            with torch.no_grad():
                non_target = torch.ones_like(outputs, dtype=torch.bool).scatter_(1, labels.unsqueeze(1), False)
                low_prob = outputs.softmax(dim=-1).le(low_prob_threshold) & non_target
                low_prob_sum += low_prob.sum() / non_target.sum()
                logit_std_sum += outputs.std(dim=-1).mean()
                train_correct += (outputs.argmax(dim=1) == labels).sum()
            seen += labels.size(0)
            batches += 1
            # print("{} step loss is {}".format(i, loss.item()))
        if scheduler is not None:
            scheduler.step()
        train_acc = round(100 * train_correct.item() / seen, 2)
        low_prob_ratio = round(low_prob_sum.item() / batches, 4)
        logit_std = round(logit_std_sum.item() / batches, 3)
        val_acc = round(run_test(valloader), 2)
        # test accuracy is only recorded; model selection uses the validation set
        test_acc = round(run_test(testloader), 2)
        if val_acc > best_val_acc:
            best_val_acc, best_epoch, test_acc_at_best = val_acc, epoch, test_acc
            save_model(net, model_path)
        L2 = L2_reg(net.parameters())
        acc.append([epoch, val_acc, test_acc, round(running_loss, 2), L2, train_acc, low_prob_ratio, logit_std])
        print("{} epoch train acc {}, val acc {}, test acc {}, L2 {}, low prob ratio {}, logit std {}".format(
            epoch, train_acc, val_acc, test_acc, L2, low_prob_ratio, logit_std), flush=True)
    print('Finished Training, best val acc {} at epoch {}, test acc {}'.format(best_val_acc, best_epoch, test_acc_at_best))
    result_file = os.path.join(result_dir, "{}.csv".format(str(t)))
    pd.DataFrame(acc).to_csv(result_file, header=["epoch", "val_acc", "test_acc", "training_loss", "L2", "train_acc",
                                                  "low_prob_ratio", "logit_std"], index=False)
    summary.append([t, best_epoch, best_val_acc, test_acc_at_best])
    pd.DataFrame(summary).to_csv(os.path.join(result_dir, "summary.csv"),
                                 header=["run", "best_epoch", "best_val_acc", "test_acc"], index=False)
    del net, optimizer
