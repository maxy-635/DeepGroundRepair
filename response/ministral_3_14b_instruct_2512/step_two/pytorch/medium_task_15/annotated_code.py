import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial feature extraction
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu1 = nn.ReLU(inplace=True)

        # Global average pooling and fully connected layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)

        # 1x1 convolution and average pooling for dimensionality reduction
        self.conv1x1 = nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # Final classification layer
        self.fc_final = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Initial feature extraction
        x_conv = self.conv1(x)  # output tensor shape: (1, 64, 32, 32)
        x_bn = self.bn1(x_conv)  # output tensor shape: (1, 64, 32, 32)
        x_relu = self.relu1(x_bn)  # output tensor shape: (1, 64, 32, 32)

        # Global average pooling and fully connected layers
        x_pool = self.adaptive_pool(x_relu)  # output tensor shape: (1, 64, 1, 1)
        x_pool_flat = torch.flatten(x_pool, 1)  # output tensor shape: (1, 64)
        x_fc1 = self.fc1(x_pool_flat)  # output tensor shape: (1, 64)
        x_fc2 = self.fc2(x_fc1)  # output tensor shape: (1, 64)

        # Reshape and multiply with initial features
        x_fc2_reshaped = x_fc2.view(-1, 64, 1, 1)  # output tensor shape: (1, 64, 1, 1)
        x_weighted = torch.mul(x_relu, x_fc2_reshaped)  # output tensor shape: (1, 64, 32, 32)
        x_concat = torch.cat([x_relu, x_weighted], dim=1)  # output tensor shape: (1, 128, 32, 32)

        # Dimensionality reduction and downsampling
        x_conv1x1 = self.conv1x1(x_concat)
        x_pool_final = self.avg_pool(x_conv1x1)

        # Flatten and final classification
        x_flat = torch.flatten(x_pool_final, 1)
        x_out = self.fc_final(x_flat)

        return x_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)