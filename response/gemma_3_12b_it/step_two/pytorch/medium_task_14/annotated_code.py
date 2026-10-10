import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )

        # Parallel Branch
        self.parallel_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features='mask', out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Path 1: Sequential Blocks
        out1 = self.block1(x)  # output tensor shape: (1, 64, 32, 32)
        out2 = self.block2(out1)  # output tensor shape: (1, 128, 32, 32)
        out3 = self.block3(out2)  # output tensor shape: (1, 256, 32, 32)

        # Path 2: Parallel Branch
        out_parallel = self.parallel_conv(x)  # output tensor shape: (1, 128, 32, 32)

        # Concatenate outputs
        combined_output = torch.cat((out3, out_parallel), dim=1)  # output tensor shape: (1, 384, 32, 32)

        # Flatten
        combined_output = combined_output.view(combined_output.size(0), -1)  # output tensor shape: (1, 393216)

        # Fully Connected Layers
        x = torch.relu(self.fc1(combined_output))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)