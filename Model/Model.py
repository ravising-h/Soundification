import torch
import torch.nn as nn
class NeuralNet(nn.Module):
    def __init__(self, num_classes = 5):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(128, 128) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,num_classes)
        self.soft = nn.Softmax()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.soft(out)
        return out