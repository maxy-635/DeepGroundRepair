import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.conv_branch = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Main pathway
        x = self.conv1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv2_1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv2_2(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        # Branch pathway
        branch = self.conv_branch(x)

        # Fuse pathways
        x = torch.add(x, branch)

        # Global average pooling, flattening, and linear layer
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = nn.Flatten()(x)
        x = nn.Linear(64 * 64, 10)(x)  # Assuming 10 output classes for MNIST
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)