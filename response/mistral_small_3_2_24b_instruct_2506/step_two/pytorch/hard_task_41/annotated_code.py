import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three parallel paths with different pooling scales
        self.avg_pool1 = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.avg_pool4 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        self.dropout = nn.Dropout(p=0.5)

        # Fully connected layer and reshaping for Block 2
        self.fc1 = nn.Linear(in_features='mask', out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

        # Block 2: Multiple branch connections
        self.conv1x1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3x3_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3x3_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.avg_pool_branch = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1x1_branch = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Block 1 processing
        x1 = self.avg_pool1(x)  # output tensor shape: (1, 1, 28, 28)
        x2 = self.avg_pool2(x)  # output tensor shape: (1, 1, 14, 14)
        x4 = self.avg_pool4(x)  # output tensor shape: (1, 1, 7, 7)

        x1_flat = torch.flatten(x1, start_dim=1)  # output tensor shape: (1, 784)
        x2_flat = torch.flatten(x2, start_dim=1)  # output tensor shape: (1, 196)
        x4_flat = torch.flatten(x4, start_dim=1)  # output tensor shape: (1, 49)

        x1_drop = self.dropout(x1_flat)  # output tensor shape: (1, 784)
        x2_drop = self.dropout(x2_flat)  # output tensor shape: (1, 196)
        x4_drop = self.dropout(x4_flat)  # output tensor shape: (1, 49)

        x_concat = torch.cat((x1_drop, x2_drop, x4_drop), dim=1)  # output tensor shape: (1, 1029)

        # Fully connected layer and reshaping
        x_fc = self.fc1(x_concat)
        x_reshape = torch.reshape(x_fc, (x_fc.size(0), 1, 8, 16))

        # Block 2 processing
        # Branch 1: 1x1 convolution
        branch1 = self.conv1x1_1(x_reshape)

        # Branch 2: 1x1 convolution, 3x3 convolution
        branch2 = self.conv1x1_2(x_reshape)
        branch2 = self.conv3x3_1(branch2)

        # Branch 3: 1x1 convolution, 3x3 convolution, 3x3 convolution
        branch3 = self.conv1x1_2(x_reshape)
        branch3 = self.conv3x3_1(branch3)
        branch3 = self.conv3x3_2(branch3)

        # Branch 4: average pooling, 1x1 convolution
        branch4 = self.avg_pool_branch(x_reshape)
        branch4 = self.conv1x1_branch(branch4)

        # Concatenate all branches
        x_branches = torch.cat((branch1, branch2, branch3, branch4), dim=1)

        # Final fully connected layers
        x_flat = torch.flatten(x_branches, start_dim=1)
        x_out = self.fc2(x_flat)

        return x_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input shape for MNIST
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)