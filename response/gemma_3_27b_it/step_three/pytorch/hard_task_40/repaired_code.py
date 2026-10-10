import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Average Pooling and Flattening
        self.pool1 = nn.AvgPool2d(kernel_size=1, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=16*16*1, out_features=64)  # Assuming input is 32x32
        self.reshape = nn.Reshape(shape=(1, 16, 16))

        # Block 2: Multi-Scale Feature Extraction
        # Path 1: 1x1 Convolution
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.1)
        )

        # Path 2: 1x1 + 3x3 + 3x3 Convolution
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.1)
        )

        # Path 3: 1x1 + 3x3 Convolution
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.1)
        )

        # Path 4: Average Pooling + 1x1 Convolution
        self.path4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.1)
        )
        
        # Concatenation and Fully Connected Layers
        self.concat = nn.Sequential(
            nn.Conv2d(in_channels=32*4, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        self.fc2 = nn.Linear(in_features=64*4*4, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Block 1
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.reshape(x)

        # Block 2
        path1 = self.path1(x)
        path2 = self.path2(x)
        path3 = self.path3(x)
        path4 = self.path4(x)

        x = torch.cat([path1, path2, path3, path4], dim=1)
        x = self.concat(x)
        x = self.flatten(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)