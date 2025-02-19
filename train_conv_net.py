# -*- coding: utf-8 -*-
"""CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LH_61JdbRgmLjGwKmoEWuWSOrAmcduvq
"""

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

# from utils import AverageMeter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 10
hidden_size = 100
num_classes = 4
num_epochs = 20
learning_rate = 0.001

#other Parameter
train_start=201
train_end=1001
validate_start=101
validate_end=201
train_size=60000
test_size=10000

#Train Data Loader
trans_img = transforms.Compose([transforms.ToTensor()])
train_dataset = FashionMNIST("./data/", train=True, transform=trans_img, download=True)
trainloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

#Test data loader
test_dataset = FashionMNIST("./data/", train=False, transform=trans_img, download=True)
testloader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

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

#create the model object and move it to GPU
model = LeNet(10).to(device)

# Loss and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

#Train the model
loss_vec = []
for i in tqdm(range(num_epochs)):
    model.train()
    avg_loss=0
    j=0
    for batch_idx, (img, target) in enumerate(trainloader):
        img = Variable(img).to(device)
        target = Variable(target).to(device)

        # Zero out the gradients
        optimizer.zero_grad()

        # Forward Propagation
        out = model(img)
        loss = F.cross_entropy(out, target)

        # backward propagation
        loss.backward()
        avg_loss+=loss
        j=j+1
        # Update the model parameters
        optimizer.step()
    avg_loss=avg_loss/j;
    print(avg_loss)
    loss_vec.append(avg_loss)

plt.figure()
plt.plot(loss_vec)
plt.title("training-loss-CNN")
plt.savefig("./images/training_CNN.jpg")

torch.save(model.state_dict(), "./models/CNN4.pt")

#evaluation of model
model.eval()
avg_loss = 0
j=0

y_gt = []
y_pred_label = []

for batch_idx, (img, y_true) in enumerate(testloader):
    img = Variable(img).to(device)
    y_true = Variable(y_true).to(device)
    y_gt.append(y_true)
    out = model(img)
    y_pred_label_tmp = torch.argmax(out, dim=1)
    y_pred_label.append(y_pred_label_tmp)
    loss = F.cross_entropy(out, y_true)
    print(loss.item())
    avg_loss+=loss.item()
    j=j+1
print(avg_loss/j)

correct=0
for i in range(10):
  for j in range(len(y_gt[i])):
    if y_gt[i][j].item()==y_pred_label[i][j].item():
      correct+=1
print(correct/10000)
