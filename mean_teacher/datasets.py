import torchvision.transforms as transforms

from . import data
from .utils import export
import os

@export
def imagenet():
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    train_transformation = data.TransformTwice(transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data-local/images/ilsvrc2012/',
        'num_classes': 1000
    }


@export
def cifar10():
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])
    train_transformation = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    if os.path.isdir("/scratch/datasets/cifar/cifar10/by-image/"):
        datadir = "/scratch/datasets/cifar/cifar10/by-image/"
    elif os.path.isdir("data-local/images/cifar/cifar10/by-image"):
        datadir = "data-local/images/cifar/cifar10/by-image"
    print("Using CIFAR-10 from", datadir)

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': datadir,
        'num_classes': 10
    }



@export
def cifar100():
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616]) # should we use different stats - do this
    train_transformation = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    if os.path.isdir("/scratch/datasets/cifar/cifar100/by-image/"):
        datadir = "/scratch/datasets/cifar/cifar100/by-image/"
    elif os.path.isdir("data-local/images/cifar/cifar100/by-image"):
        datadir = "data-local/images/cifar/cifar100/by-image"
    print("Using CIFAR-100 from", datadir)

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': datadir,
        'num_classes': 100
    }






















