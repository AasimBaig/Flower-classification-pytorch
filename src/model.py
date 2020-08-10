import config
import torch
import torch.nn.functional as F
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layer_1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=2),
                                          nn.Dropout2d(p=0.1)
                                          )
        self.conv_layer_2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=4),
                                          nn.Dropout2d(p=0.05)
                                          )
        self.conv_layer_3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
                                          nn.ReLU(),
                                          nn.MaxPool2d(kernel_size=5),
                                          nn.Dropout2d(p=0.05)
                                          )
        self.fully_connected_layer_1 = nn.Linear(
            64 * (int)((config.IMAGE_HEIGHT/(2*4*5)) * (config.IMAGE_WIDTH/(2*4*5))), 1000)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)
        self.fully_connected_layer_2 = nn.Linear(1000, 5)

    def forward(self, x):
        out = self.conv_layer_1(x)
        out = self.conv_layer_2(out)
        out = self.conv_layer_3(out)
        out = out.view(out.size(0), -1)
        out = self.fully_connected_layer_1(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fully_connected_layer_2(out)
        return out
