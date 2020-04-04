import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import MLP


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_train = datasets.CIFAR10('/data', train=True, download=True, transform=transform)
    train_loader = DataLoader(data_train, num_workers=4, batch_size=32)
    data_test = datasets.CIFAR10('/data', train=True, download=True, transform=transform)
    test_loader = DataLoader(data_test, num_workers=4, batch_size=16)

    # configure network
    net = MLP(3 * 32 * 32, 10)
    net.to(device)
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # configure tensorboard writer
    name = '.'.join(str(datetime.datetime.now()).split(':'))
    writer = SummaryWriter(f'./logs/{name}')

    epochs = 5
    running_loss = 0.0
    for epoch in range(epochs):
        net.train()
        # i = 0
        for i,data, labels in enumerate(train_loader):
            # i += 1
            data, labels = data.to(device), labels.to(device)
            data = data.view(-1, 3 * 32 * 32)
            writer.add_graph(net, data)
            optimizer.zero_grad()
            net_out = net(data)
            loss = criterion(net_out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 400 == 399:  # print every 400 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 400))
                running_loss = 0.0
                avg_loss, acc = evaluate(net, test_loader, criterion, device)
                print(f'avg_loss: {avg_loss}, acc: {acc}')
                writer.add_scalar('acc', acc, epoch * len(train_loader) + i)
                writer.add_scalar('avg_loss', avg_loss, epoch * len(train_loader) + i)
                writer.add_scalar('learning rate', scheduler.get_lr()[0], epoch * len(train_loader) + i)
        scheduler.step()
    print('Finished Training')
    # save model
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)


if __name__ == '__main__':
    main()
