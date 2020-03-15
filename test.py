import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from model import MLP
import train
criterion = nn.CrossEntropyLoss()
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
data_test = datasets.CIFAR10('/data', train=False, download=True, transform=transform)

net = MLP(3 * 32 * 32, 10)
PATH = './cifar_net.pth'
net.load_state_dict(torch.load(PATH))
print(net)
test_loader = DataLoader(data_test, num_workers=0, batch_size=4)  # error with num_workers != 0
criterion = nn.CrossEntropyLoss()
# accuracy computation
all_labels_number = 0
all_correct_labels_number = 0
for data, label in test_loader:
    # print(data,label)
    net_out = net(data)
    _, label_pred = torch.max(net_out, 1)
    labels_in_batch = label.shape[0]
    correct_labels_in_batch = (label_pred == label).sum().item()
    all_labels_number += labels_in_batch
    all_correct_labels_number += correct_labels_in_batch
    # print(all_labels_number)
print(f'accuracy: {all_correct_labels_number/all_labels_number}') # 0.5237

