import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        # Block 2
        self.path1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.bn_path1 = nn.BatchNorm2d(64)
        self.relu_path1 = nn.ReLU(inplace=True)

        self.path2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_path2 = nn.BatchNorm2d(64)
        self.relu_path2 = nn.ReLU(inplace=True)
        self.avgpool2 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_path2_after_pool = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.bn_path2_after_pool = nn.BatchNorm2d(64)
        self.relu_path2_after_pool = nn.ReLU(inplace=True)

        self.path3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.bn_path3 = nn.BatchNorm2d(64)
        self.relu_path3 = nn.ReLU(inplace=True)
        self.path3_1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.bn_path3_1 = nn.BatchNorm2d(64)
        self.relu_path3_1 = nn.ReLU(inplace=True)
        self.path3_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_path3_2 = nn.BatchNorm2d(64)
        self.relu_path3_2 = nn.ReLU(inplace=True)

        self.path4 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.bn_path4 = nn.BatchNorm2d(64)
        self.relu_path4 = nn.ReLU(inplace=True)
        self.conv_path4_after_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_path4_after_conv = nn.BatchNorm2d(64)
        self.relu_path4_after_conv = nn.ReLU(inplace=True)
        self.path4_1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.bn_path4_1 = nn.BatchNorm2d(64)
        self.relu_path4_1 = nn.ReLU(inplace=True)
        self.path4_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn_path4_2 = nn.BatchNorm2d(64)
        self.relu_path4_2 = nn.ReLU(inplace=True)


        # Flatten and Linear
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64 * 8, 10)  # Output layer for CIFAR-10

    def forward(self, x):
        # Block 1
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))

        # Block 2
        x_path1 = self.relu_path1(self.bn_path1(self.path1(x)))
        x_path2 = self.relu_path2(self.bn_path2(self.conv_path2_after_pool(self.avgpool2(x))))
        x_path3 = self.relu_path3(self.bn_path3(self.path3(x)))
        x_path3_1 = self.relu_path3_1(self.bn_path3_1(self.path3_1(x_path3)))
        x_path3_2 = self.relu_path3_2(self.bn_path3_2(self.path3_2(x_path3)))
        x_path3 = self.concatenate([x_path3_1, x_path3_2])

        x_path4 = self.relu_path4(self.bn_path4(self.path4(x)))
        x_path4_1 = self.relu_path4_1(self.bn_path4_1(self.path4_1(x_path4)))
        x_path4_2 = self.relu_path4_2(self.bn_path4_2(self.path4_2(x_path4)))
        x_path4 = self.concatenate([x_path4_1, x_path4_2])

        # Concatenate paths
        x = self.concatenate([x_path1, x_path2, x_path3, x_path4])

        # Flatten and Linear
        x = self.flatten(x)
        x = self.linear(x)

        return x