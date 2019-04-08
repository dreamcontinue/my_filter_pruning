import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def cifar100DataLoader(is_train=True,batch_size=64,shuffle=True,workers=2):
    if is_train:
        trans = [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, padding=4),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[n / 255.
                                            for n in [129.3, 124.1, 112.4]],
                                      std=[n / 255. for n in [68.2, 65.4, 70.4]])]
        trans = transforms.Compose(trans)
        train_set = datasets.CIFAR100('data', train=True, transform=trans, download=True)
        loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=shuffle,num_workers=workers, pin_memory=True)
    else:
        trans = [transforms.ToTensor(),
                 transforms.Normalize(mean=[n / 255.
                                            for n in [129.3, 124.1, 112.4]],
                                      std=[n / 255. for n in [68.2, 65.4, 70.4]])]
        trans = transforms.Compose(trans)
        test_set = datasets.CIFAR100('data', train=False, transform=trans)
        loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=shuffle,num_workers=workers, pin_memory=True)
    return loader

def cifar10DataLoader(is_train=True,batch_size=64,shuffle=True,workers=2):
    if is_train:
        trans = [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, padding=4),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[x / 255 for x in [125.3, 123.0, 113.9]],
                                      std=[x / 255 for x in [63.0, 62.1, 66.7]])]
        trans = transforms.Compose(trans)
        train_set = datasets.CIFAR10('data', train=True, transform=trans, download=True)
        loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=shuffle,num_workers=workers, pin_memory=True)
    else:
        trans = [transforms.ToTensor(),
                 transforms.Normalize(mean=[x / 255 for x in [125.3, 123.0, 113.9]],
                                      std=[x / 255 for x in [63.0, 62.1, 66.7]])]
        trans = transforms.Compose(trans)
        test_set = datasets.CIFAR10('data', train=False, transform=trans)
        loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=shuffle,num_workers=workers, pin_memory=True)
    return loader


def ImageNetDataLoader(is_train=True,batch_size=64,shuffle=True):
    return