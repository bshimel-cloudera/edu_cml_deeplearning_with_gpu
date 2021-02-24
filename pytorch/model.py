#
#  The Model to be loaded into the training code
#
#
import torch.nn as nn


class FashionCNN(nn.Module):
    """FashionCNN Model

    A Pytorch model to run classification on the Fashion-MNist Dataset

    """
    
    def __init__(self):
        """Pytorch Model Initialisation

        This consists of a simple network with 2 Convolution Layers to encode the images 
        followed by three Linear layers to produce the final classification

        """
        super(FashionCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        """Pytorch Model Forward function

        Takes a sample batch of images and feeds it through the network
        Pytorch uses N x C x H x W

        Arguments:
             x (Torch.tensor): a valid batch of images 

        """
    
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out