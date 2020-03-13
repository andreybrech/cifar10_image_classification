import torch
import torch.nn as nn
from torchvision import datasets

def main():
    data_train = datasets.CIFAR10('./data',train=True,download=True)
    data_test = datasets.CIFAR10('./data',train=True,download=True)

if __name__ == '__main__':
    main()