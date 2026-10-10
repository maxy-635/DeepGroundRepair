import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: Depthwise separable convolution + 1x1 convolution + Dropout
        self.branch1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Depthwise separable convolution
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, stride=1),  # 1x1 convolution
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)  # Dropout layer
        )

        # Branch 2: Depthwise separable convolution + 1x1 convolution + Dropout
        self.branch2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Depthwise separable convolution
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, stride=1),  # 1x1 convolution
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)  # Dropout layer
        )

        # Branch 3: Depthwise separable convolution + 1x1 convolution + Dropout
        self.branch3 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Depthwise separable convolution
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, stride=1),  # 1x1 convolution
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)  # Dropout layer
        )

        # Concatenate the outputs of the three branches
        self.concatenate = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1, stride=1),  # Concatenate 3 branches (32*3)
            nn.ReLU(inplace=True)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10)  # Assuming 10 classes for MNIST
        )

    def forward(self, x):
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)

        # Concatenate the outputs of the three branches
        concatenated = torch.cat((branch1_out, branch2_out, branch3_out), dim=1)
        concatenated = self.concatenate(concatenated)

        # Fully connected layers
        x = self.fc_layers(concatenated)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)