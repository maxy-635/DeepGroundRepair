import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block
        self.pool1 = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        self.flatten = nn.Flatten()

        # Fully connected layer and reshape
        self.fc1 = nn.Linear(in_features='mask', out_features=64)
        
        # Second block
        self.split = lambda x: torch.split(x, x.shape[-1] // 4, dim=-1) # Splits along the last dimension
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )
        
        self.flatten2 = nn.Flatten()
        self.fc2 = nn.Linear(in_features=16 * 16 * 4, out_features=10) # Assuming input size to second block is 16x16 after reshaping.

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # First block
        x1 = self.pool1(x)  # output tensor shape: (1, 1, 28, 28)
        x2 = self.pool2(x)  # output tensor shape: (1, 1, 14, 14)
        x3 = self.pool3(x)  # output tensor shape: (1, 1, 7, 7)
        x1 = self.flatten(x1)  # output tensor shape: (1, 784)
        x2 = self.flatten(x2)  # output tensor shape: (1, 196)
        x3 = self.flatten(x3)  # output tensor shape: (1, 49)
        x = torch.cat([x1, x2, x3], dim=1)  # output tensor shape: (1, 1029)

        # Fully connected layer and reshape
        x = self.fc1(x)
        x = x.reshape(x.shape[0], 1, 16, 4)
        
        # Second block
        x_split = self.split(x)
        x1 = self.conv1(x_split[0])
        x2 = self.conv2(x_split[1])
        x3 = self.conv3(x_split[2])
        x4 = self.conv4(x_split[3])
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.flatten2(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Dummy input tensor (batch_size=1, channels=1, height=28, width=28)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)