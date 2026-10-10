import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_gap = nn.AdaptiveAvgPool2d(1)
        self.block1_fc1 = nn.Linear(in_features=3, out_features=1024, bias=True)
        self.block1_fc2 = nn.Linear(1024, 1024, bias=True)

        # Block 2
        self.block2_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.block2_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.block2_maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Classification
        self.classifier_fc1 = nn.Linear(64 * 8 * 8, 128, bias=True)
        self.classifier_fc2 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1
        x = self.block1_gap(x)  # output tensor shape: (1, 3, 1, 1)
        x = x.view(x.size(0), -1)  # output tensor shape: (1, 3)
        x = torch.relu(self.block1_fc1(x))
        x = torch.relu(self.block1_fc2(x))
        x = x.view(x.size(0), 1024, 1, 1)

        # Block 2
        x2 = F.relu(self.block2_conv1(x))
        x2 = F.relu(self.block2_conv2(x2))
        x2 = self.block2_maxpool(x2)

        # Fusion
        x = x + x2

        # Classification
        x = x.view(x.size(0), -1)
        x = torch.relu(self.classifier_fc1(x))
        x = self.classifier_fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Dummy input tensor (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)