import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    """ A implementation of the AlexNet architecture from the paper 'ImageNet 
        Classification with Deep Convolutional Neural Networks' """
    def __init__(self):
        super().__init__()
        # The paper specifies that this layer should be stride 4 however this does not work with 
        # the input dimensions, I work under the assumption that the definition of stride used 
        # in the Alexnet paper and the pytorch library differ.
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=5, padding_mode='reflect') 
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, padding_mode='reflect')
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.fc6 = nn.Linear(43264,4096)
        self.fc7 = nn.Linear(4096,4096)
        self.fc8 = nn.Linear(4096,1000)
        self.pool = nn.MaxPool2d(3,2)
        self.drop = nn.Dropout(0.5)
        self.lrn = nn.LocalResponseNorm(size=5,alpha=10e-4,beta=0.75,k=2.0)

    def forward(self, x):
        x = self.lrn(self.pool(F.relu(self.conv1(x))))
        x = self.lrn(self.pool(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1,43264)
        x = F.relu(self.drop(self.fc6(x)))
        x = F.relu(self.drop(self.fc7(x)))
        x = self.fc8(x)
        return x

