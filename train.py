import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from model import MLP


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_train = datasets.CIFAR10('/data', train=True, download=True,transform=transform)
    data_test = datasets.CIFAR10('/data', train=True, download=transform)
    train_loader = DataLoader(data_train, num_workers=4, batch_size=32)
    test_loader = DataLoader(data_test, num_workers=4, batch_size=16)

    net = MLP(3*32*32,10)
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    epochs=5
    running_loss = 0.0
    for epoch in range(epochs):
        i = 0
        for data,labels in train_loader:
            i += 1
            data = data.view(-1,3*32*32)
            optimizer.zero_grad()
            net_out = net(data)
            loss = criterion(net_out,labels)
            # print(loss)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
    print('Finished Training')
    #save model
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
if __name__ == '__main__':
    main()

