import datetime
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import MLP

# training settings
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size_train', type=int, default=16,
                    help="batch size for training (default:16)")
parser.add_argument('--batch_size_test', type=int, default=16,
                    help="batch size for testing (default:16)")
parser.add_argument('--epochs', type=int, default=5,
                    help="epochs number for training (default:5)")
parser.add_argument('--lr', type=int, default=0.01,
                    help="learning rate (default:0.01)")
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--lr_sheduler', action='store_true', default=True,
                    help='enables lr_sheduler training')


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
    print(args)
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
    net = MLP(3 * 32 * 32, 10)
    net.to(device)
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)

    # configure tensorboard writer
    name = '.'.join(str(datetime.datetime.now()).split(':'))
    writer = SummaryWriter(f'./logs/{name}')

    epochs = args.epochs
    running_loss = 0.0
    for epoch in range(epochs):
        net.train()
        i = 0
        for data, labels in train_loader:
            i += 1
            data, labels = data.to(device), labels.to(device)
            data = data.view(-1, 3 * 32 * 32)
            writer.add_graph(net, data)
            optimizer.zero_grad()
            net_out = net(data)
            loss = criterion(net_out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 400 == 0:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 400))
                running_loss = 0.0
                avg_loss, acc = evaluate(net, test_loader, criterion, device)
                print(f'avg_loss: {avg_loss}, acc: {acc}')
                writer.add_scalar('acc', acc, epoch * len(train_loader) + i)
                writer.add_scalar('avg_loss', avg_loss, epoch * len(train_loader) + i)
    print('Finished Training')
    # save model
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)


if __name__ == '__main__':
    main()
