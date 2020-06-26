import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv1 = nn.Conv2d(3,96,11,4)
        self.conv2 = nn.Conv2d(96,256,5,1,padding=2)
        self.conv3 = nn.Conv2d(256,384,3,1,padding=1)
        self.conv4 = nn.Conv2d(384,384,3,1,padding=1)
        self.conv5 = nn.Conv2d(384,256,3,1,padding=1)
        self.fc1 = nn.Linear(256 * 6*6, 4096 )
        self.fc2 = nn.Linear(4096, 4096 )
        self.fc3 = nn.Linear(4096, 1000 )
    

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 3,stride=2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,3,stride=2)

        # x = F.max_pool2d(F.relu(self.conv2(x)), 3,stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 3,stride=2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    
net = AlexNet()
print(net)
dummyInput = torch.rand(1,3,227,227)
net.forward(dummyInput)