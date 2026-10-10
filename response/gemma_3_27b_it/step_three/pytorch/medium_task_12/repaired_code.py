import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        block1 = nn.Sequential(  # output tensor shape: (unknown)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True)
        )

        # Block 2
        block2 = nn.Sequential(  # output tensor shape: (unknown)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True)
        )

        # Block 3
        block3 = nn.Sequential(  # output tensor shape: (unknown)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        
        self.blocks = nn.ModuleList([block1, block2, block3])

        # Concatenation and Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=114688, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        outputs = []  # output tensor shape: (unknown)
        for block in self.blocks:  # output tensor shape: (unknown)
            x = block(x)  # output tensor shape: (1, 64, 32, 32)
            outputs.append(x)
        
        x = torch.cat(outputs, dim=1)  # output tensor shape: (1, 112, 32, 32)
        x = self.flatten(x)  # output tensor shape: (1, 114688)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)