import torch.nn as nn
a = nn.Module

class MLP(nn.Module):

    def __init__(self, input_size, num_classes):
        super(MLP,self).__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_size,32*32)
        self.fc2 = nn.Linear(32*32,200)
        self.fc3 = nn.Linear(200,num_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    pass