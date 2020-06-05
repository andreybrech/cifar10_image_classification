import datetime
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from operator import methodcaller
from torchvision import datasets
from torch.utils.data import DataLoader


def evaluate(model, test_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        sum_loss = 0
        correct_labels_num = 0
        all_labels_num = 0
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            out = model(data)
            loss = criterion(out, label)
            # print(out.data)
            _, label_pred = torch.max(out.data, 1)
            correct_labels_num += (label_pred == label).sum().item()
            sum_loss += loss.item()
            all_labels_num += label.shape[0]
        avg_loss = sum_loss / all_labels_num
        acc = correct_labels_num / all_labels_num
    model.train()
    return avg_loss, acc


def make_data(args, data_path='/data', num_workers=4):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    methodcaller_train = methodcaller(args.dataset_name, root=data_path, train=True, download=True, transform=transform)
    methodcaller_test = methodcaller(args.dataset_name, root=data_path, train=False, download=True, transform=transform)
    data_train = methodcaller_train(datasets)
    train_loader = DataLoader(data_train, num_workers=num_workers, batch_size=args.batch_size_train)
    data_test = methodcaller_test(datasets)
    test_loader = DataLoader(data_test, num_workers=num_workers, batch_size=args.batch_size_test)
    # datasets.CIFAR10()
    return train_loader, test_loader


def load_model_from_pretrained(args, net):
    epoch = args.from_pretrained_epoch
    eval_num = args.from_pretrained_eval
    if len(os.listdir(f'./models/{args.network_type}')) == 0:
        print('No pretrained model in model folder. Weights are not loaded')
    if args.from_pretrained_epoch == -1:
        if f'{args.dataset_name}_{args.network_type}_final.pth' in os.listdir(f'./models/{args.network_type}'):
            PATH = f'./models/{args.network_type}/{args.dataset_name}_{args.network_type}_final.pth'
        else:
            weights_name = sorted([x for x in os.listdir(f'./models/{args.network_type}') if '.pth' in x])[-1]
            PATH = f'./models/{args.network_type}/' + weights_name

    else:
        epoch = args.from_pretrained_epoch
        eval_num = args.from_pretrained_eval
        if eval_num > -1:
            weights_name_arr = sorted([x for x in os.listdir(f'./models/{args.network_type}') if ('.pth' in x and f'epoch_{epoch}' in x)])
            assert len(weights_name_arr) > 0, 'No weights from this epoch'
            weights_name = weights_name_arr[-1]
            PATH = f'./models/{args.network_type}/' + weights_name
        else:
            assert  eval_num == -1, 'No such eval num from this epoch'
            weights_name_arr = sorted([x for x in os.listdir(f'./models/{args.network_type}') if
                                       ('.pth' in x and f'epoch_{epoch}' in x and f'eval_{eval_num}' in x)])
            assert len(weights_name_arr) > 0, 'No weights from this epoch'
            weights_name = weights_name_arr[-1]
            PATH = f'./models/{args.network_type}/' + weights_name

    print(f'loaded model from {PATH}')
    net.load_state_dict(torch.load(PATH))
    return net
