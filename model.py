import torch.nn as nn

a = nn.Module


class MLP(nn.Module):

    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_size, 32 * 32)
        self.fc2 = nn.Linear(32 * 32, 200)
        self.fc3 = nn.Linear(200, num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):

    def __init__(self, input_size, num_classes):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 6, 6)
        self.conv2 = nn.Conv2d(6, 18, 6)
        self.conv3 = nn.Conv2d(18, 18, 10)
        self.fc1 = nn.Linear(18 * 13 * 13, 200)
        self.fc2 = nn.Linear(200, num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    pass
