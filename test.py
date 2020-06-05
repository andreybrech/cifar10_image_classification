import argparse
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from model import MLP
import train

import datetime
import argparse
import os
from operator import attrgetter
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import model
from utils import evaluate, make_data

def main(parser):
    args = parser.parse_args()
    criterion = nn.CrossEntropyLoss()
    _, test_loader = make_data(args, data_path='/data', num_workers=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    getter = attrgetter(args.network_type)
    nn_type = getter(model)
    net = nn_type(input_size=3*32*32, num_classes=10)
    net.to(device)
    print(net)

    PATH = './cifar_net.pth'
    assert len(os.listdir(f'./models/{args.network_type}')) != 0, 'No pretrained model in model folder'
    if f'{args.dataset_name}_{args.network_type}_final.pth' in  os.listdir(f'./models/{args.network_type}'):
        PATH = f'./models/{args.network_type}/{args.dataset_name}_{args.network_type}_final.pth'
    else:
        weights_name = sorted([x for x in os.listdir(f'./models/{args.network_type}') if '.pth' in x])[-1]
        PATH = f'./models/{args.network_type}/' + weights_name
    print()
    net.load_state_dict(torch.load(PATH))
    print(net)
    criterion = nn.CrossEntropyLoss()
    avg_loss, acc = evaluate(net, test_loader, criterion, device)
    print(f'avg_loss: {avg_loss}, acc: {acc} on model from {PATH}')
if __name__ == '__main__':
    # training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', choices=['CIFAR10'], default='CIFAR10',
                        help='choose dataset name. Available:[CIFAR10]')
    parser.add_argument('--batch_size_train', type=int, default=16,
                        help="batch size for training (default:16)")
    parser.add_argument('--batch_size_test', type=int, default=16,
                        help="batch size for testing (default:16)")
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--network_type', choices=['CNN', 'MLP'], default='MLP',
                        help='choose model type name. Available: [CNN, MLP]')
    parser.add_argument('--info_times_per_epoch',  type=int, default=4,
                        help='chose frequency of info per epoch ')
    parser.add_argument('--path_to_model', type=int, default=4,
                        help='chose frequency of info per epoch ')
    main(parser)