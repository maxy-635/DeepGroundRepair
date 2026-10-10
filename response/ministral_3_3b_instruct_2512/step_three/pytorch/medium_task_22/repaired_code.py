import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 3x3 convolutions
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Branch 2: 1x1 convolution followed by two 3x3 convolutions
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Branch 3: Max pooling
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Multi-scale feature fusion block
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        branch1_out = self.branch1(x)  # output tensor shape: (1, 64, 16, 16)
        branch2_out = self.branch2(x)  # output tensor shape: (1, 64, 16, 16)
        branch3_out = self.branch3(x)  # output tensor shape: (1, 3, 16, 16)

        # Concatenate outputs from all branches
        fused_features = torch.cat((branch1_out, branch2_out, branch3_out), dim=1)  # output tensor shape: (1, 131, 16, 16)
        fused_features = self.fusion(fused_features)

        # Flatten and pass through fully connected layers
        x = self.fc_layers(fused_features)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor for CIFAR-10 (batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (1, 10)