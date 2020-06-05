import datetime
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import model
from operator import attrgetter
# from model import MLP, CNN



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


def main():
    args = parser.parse_args()
    print(f'args:{args}')
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    # load dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_train = datasets.CIFAR10('/data', train=True, download=True, transform=transform)
    train_loader = DataLoader(data_train, num_workers=4, batch_size=args.batch_size_train)
    data_test = datasets.CIFAR10('/data', train=True, download=True, transform=transform)
    test_loader = DataLoader(data_test, num_workers=4, batch_size=args.batch_size_test)

    # configure network
    getter = attrgetter(args.network_type)
    nn_type = getter(model)
    net = nn_type(3 * 32 * 32, 10)
    net.to(device)
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # configure tensorboard writer
    name = '.'.join(str(datetime.datetime.now()).split(':'))
    writer = SummaryWriter(f'./logs/{name}')

    epochs = args.epochs
    running_loss = 0.0
    info_every = int(len(train_loader) / args.info_times_per_epoch)
    eval_every = int(len(train_loader) / args.eval_times_per_epoch)
    for epoch in range(epochs):
        net.train()
        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            # print(data.shape,labels.shape)
            net_out = net(data)
            # print(net_out, labels)
            # print(net_out.shape,labels.shape)
            loss = criterion(net_out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % info_every == info_every - 1:  # print every 400 mini-batches
                print('[%d, %5d / %5d] loss: %.3f' %
                      (epoch + 1, i + 1, len(train_loader), running_loss / info_every, ))
                running_loss = 0.0
            if i % eval_every == eval_every - 1:
                avg_loss, acc = evaluate(net, test_loader, criterion, device)
                print(f'avg_loss: {avg_loss}, acc: {acc}')
                writer.add_scalar('acc', acc, epoch * len(train_loader) + i)
                writer.add_scalar('avg_loss', avg_loss, epoch * len(train_loader) + i)
                writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch * len(train_loader) + i)
                if args.save_model_on_eval:
                    PATH = f'./models/{args.network_type}/cifar_net_{args.network_type}_epoch_{epoch}_eval_{i // eval_every}.pth'
                    torch.save(net.state_dict(), PATH)

        if args.lr_sheduler:
            scheduler.step()
    avg_loss, acc = evaluate(net, test_loader, criterion, device)
    print(f'Finished Training. Avarage loss: {avg_loss}, accuracy: {acc}')
    # save model
    if args.save_model:
        PATH = './models/cifar_net.pth'
        torch.save(net.state_dict(), PATH)


if __name__ == '__main__':
    # training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size_train', type=int, default=16,
                        help="batch size for training (default:16)")
    parser.add_argument('--batch_size_test', type=int, default=16,
                        help="batch size for testing (default:16)")
    parser.add_argument('--epochs', type=int, default=5,
                        help="epochs number for training (default:5)")
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate (default:0.01)")
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='enables CUDA training')
    parser.add_argument('--lr_sheduler', action='store_true', default=False,
                        help='enables lr_sheduler training')
    parser.add_argument('--network_type', choices=['CNN', 'MLP', 'scissors'], default='MLP',
                        help="CNN; MLP")
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='enables saving model')
    # parser.add_argument('--detailed_statistics', action='store_true', default=False,
    #                     help='enables detailed statistics: accuracy, avg_loss 4 time per epoch and write it to tensorboard')
    parser.add_argument('--save_model_on_eval', action='store_true', default=False,
                        help='enables saving model on every evaluation')
                             # 'enables detailed statistics: accuracy, avg_loss 4 time per epoch and write it to tensorboard')
    parser.add_argument('--info_times_per_epoch',  type=int, default=4,
                        help='chose frequency of info per epoch ')
    parser.add_argument('--eval_times_per_epoch', type=int, default=4,
                        help='chose frequency of evaluation per epoch ')


    main()