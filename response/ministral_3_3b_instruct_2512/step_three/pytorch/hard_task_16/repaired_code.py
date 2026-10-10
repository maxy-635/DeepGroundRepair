import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Splitting and concatenating three convolutional groups
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Transition Convolution
        self.transition_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Block 2: Global Max Pooling and Weight Generation
        self.max_pool = nn.MaxPool2d(kernel_size=8, stride=8)
        self.fc1 = nn.Linear(64 * 4 * 4, 64 * 4)
        self.fc2 = nn.Linear(64 * 4, 64)
        self.relu = nn.ReLU(inplace=True)

        # Fully connected layer for classification
        self.fc = nn.Linear(64, 10)

        # Branch for direct connection
        self.branch = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1 processing
        x1 = self.block1(x)  # output tensor shape: (1, 64, 32, 32)
        x1 = x1.view(x1.size(0), -1)  # output tensor shape: (1, 65536)

        # Transition Convolution
        x2 = self.transition_conv(x1)

        # Branch processing
        branch_output = self.branch(x)

        # Global Max Pooling and weight generation
        pooled = self.max_pool(x2)
        pooled = pooled.view(pooled.size(0), -1)
        weights = self.fc1(pooled)
        weights = self.relu(weights)
        weights = self.fc2(weights)
        weights = weights.view(weights.size(0), 64, 4, 4)

        # Weighted main path output
        main_path_output = torch.matmul(weights, x2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Add outputs from main path and branch
        output = main_path_output + branch_output

        # Classification
        output = output.view(output.size(0), -1)
        return self.fc(output)

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)