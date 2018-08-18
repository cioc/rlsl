import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, h):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv1d(2, 16, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(16)
        self.conv5 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm1d(32)
        self.conv6 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.bn6 = nn.BatchNorm1d(32)

        self.fc1 = nn.Linear(128000 + 32000, 1024)
        self.fc2 = nn.Linear(1024, h + 2) #action 4 is left, action 5 is right; each other action represents the quality of picking height h for the node

    def forward(self, x):
        p0 = F.relu(self.bn1(self.conv1(x[:,0:3])))
        p0 = F.relu(self.bn2(self.conv2(p0)))
        p0 = F.relu(self.bn3(self.conv3(p0)))
        p0 = p0.view(-1, 128000)
        
        p1 = F.relu(self.bn4(self.conv4(x[:, 3:5,3,:])))
        p1 = F.relu(self.bn5(self.conv5(p1)))
        p1 = F.relu(self.bn6(self.conv6(p1)))
        p1 = p1.view(-1, 32000)

        o = torch.cat((p0, p1), 1)
        o = F.relu(self.fc1(o))
        o = self.fc2(o)
        return o
