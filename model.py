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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=6)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=6)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(128*4*4, 200)
        self.fc2 = nn.Linear(200, num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.max_pool1(self.conv1(x)))
        x = self.act(self.max_pool2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    pass
