# Task 3 CNN (Practice)
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
import rasterio
from rasterio.enums import Resampling
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=2)
image, label = train_data[0]
image.size()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 12, 5)         # new shape is (12, 28, 28)
        self.pool = nn.MaxPool2d(2, 2)           # new shape is (12, 14, 14)
        self.conv2 = nn.Conv2d(12, 24, 5)        # new shape is (24, 10, 10) -> (24, 5, 5) -> flatten -> (24 * 5 * 5)
        self.fcl = nn.Linear(24 * 5 * 5, 120)    # Fully connected: 24 * 5 * 5 to 120
        self.fcl2 = nn.Linear(120, 84)           # Fully connected: 120 to 84
        self.fcl3 = nn.Linear(84, 10)            # Fully connected: 84 to 10 (number of classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fcl(x))
        x = F.relu(self.fcl2(x))
        x = self.fcl3(x)
        return x
    
    #input -> conv1+ReLU+pool -> conv2+ReLU+pool -> flatten -> fc1+ReLU -> fc2+ReLU -> fc3
    #Output is raw logits for 10 classes
net = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(15): # loop over the dataset multiple times to train it 
    print(f"Epoch {epoch + 1}")
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    print (f"Loss: {running_loss / len(train_loader):.3f}")
torch.save(net.state_dict(), "trained_net.pth")
net = NeuralNetwork()
net.load_state_dict(torch.load("trained_net.pth"))

correct = 0
total = 0

net.eval()
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%") # accuracy is around 68%, ok for 15 epochs of training