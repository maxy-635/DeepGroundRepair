import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Branch path
        self.branch_conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Flatten and fully connected layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=12544, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Main path
        main_path = F.relu(self.conv1(x))  # output tensor shape: (1, 16, 28, 28)
        main_path = self.dropout1(main_path)  # output tensor shape: (1, 16, 28, 28)
        main_path = F.relu(self.conv2(main_path))  # output tensor shape: (1, 32, 28, 28)
        main_path = F.relu(self.conv3(main_path))  # output tensor shape: (1, 16, 28, 28)

        # Branch path
        branch_path = F.relu(self.branch_conv(x))  # output tensor shape: (1, 16, 28, 28)

        # Combine paths
        combined_path = torch.add(main_path, branch_path)  # output tensor shape: (1, 16, 28, 28)

        # Flatten and fully connected layer
        x = self.flatten(combined_path)  # output tensor shape: (1, 12544)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Dummy input tensor (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)