import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32 * 32 * 32, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (64, 3, 32, 32)
        x1 = self.conv1x1(x)  # output tensor shape: (64, 32, 32, 32)
        x2 = self.conv3x3(x)  # output tensor shape: (64, 32, 32, 32)
        x3 = self.conv5x5(x)  # output tensor shape: (64, 32, 32, 32)
        
        split_size = x1.shape[1] // 3  # output tensor shape: (1)
        x1_split = torch.split(x1, split_size, dim=1)  # output tensor shape: (unknown)
        x2_split = torch.split(x2, split_size, dim=1)  # output tensor shape: (unknown)
        x3_split = torch.split(x3, split_size, dim=1)  # output tensor shape: (unknown)
        
        concatenated = torch.cat([torch.cat([x1_split[i], x2_split[i], x3_split[i]] , dim=1) for i in range(x1.shape[0])], dim=0)  # output tensor shape: (1)
        
        x = self.flatten(concatenated)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(64, 3, 32, 32)  # Batch size 64, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)