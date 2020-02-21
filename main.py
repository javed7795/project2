import numpy as np
import pdb
import os
from tqdm import tqdm
import sys

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.datasets import FashionMNIST
from torchvision import transforms


# Device configuration
device = torch.device('cpu')

#Test data loader
trans_img = transforms.Compose([transforms.ToTensor()])
test_dataset = FashionMNIST("./data/", train=False, transform=trans_img, download=True)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#loss function
def test(model, testloader):
    model.eval()
    avg_loss=0
    j=0
    y_gt = []
    y_pred_label = []

    for batch_idx, (img, y_true) in enumerate(testloader):
        img = Variable(img)
        y_true = Variable(y_true)
        out = model(img)
        y_pred_label_tmp = torch.argmax(out, dim=1)

        loss = F.cross_entropy(out, y_true)
        avg_loss+=loss.item()
        j=j+1

        # Add the labels
        y_gt += list(y_true.numpy())
        y_pred_label += list(y_pred_label_tmp.numpy())

    return avg_loss/j, y_gt, y_pred_label



#convolutional neural network
class LeNet(nn.Module):

    def __init__(self, n_classes=10):
        emb_dim = 20
        super(LeNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 256, 1000)
        self.fc2 = nn.Linear(1000, 10)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        # out = self.fc2(out)
        out = F.log_softmax(self.fc2(out), dim=1)
        return out

# Fully connected neural network with one hidden layer
class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128 , 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.cln = nn.Linear(16 ,10)

    
    def forward(self, x):
        x = x.view(-1,784) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.log_softmax(self.cln(x), dim=1)
        return x


mlp_model = MLP(10).to(device)
cnn_model = LeNet(10).to(device)

mlp_model.load_state_dict(torch.load("./models/best_MLP.pt", map_location=torch.device('cpu')))
cnn_model.load_state_dict(torch.load("./models/best_CNN.pt", map_location=torch.device('cpu')))

loss, gt, pred = test(mlp_model, testloader)
with open("multi-layer-net.txt", 'w') as f:
    f.write("Loss on Test Data : {}\n".format(loss))
    f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
    f.write("gt_label,pred_label \n")
    for idx in range(len(gt)):
        f.write("{},{}\n".format(gt[idx], pred[idx]))

loss, gt, pred = test(cnn_model, testloader)
with open("convolution-neural-net.txt", 'w') as f:
    f.write("Loss on Test Data : {}\n".format(loss))
    f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
    f.write("gt_label,pred_label \n")
    for idx in range(len(gt)):
        f.write("{},{}\n".format(gt[idx], pred[idx]))

