import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(p=0.25)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

        # Branch path
        self.branch_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=28*28, out_features=10)

    def forward(self, x):
        # Main path
        x_main = self.conv1(x)
        x_main = self.dropout1(x_main)
        x_main = self.conv2(x_main)
        x_main = self.dropout2(x_main)
        x_main = self.conv3(x_main)

        # Branch path
        x_branch = self.branch_conv(x)

        # Combine paths
        x_combined = torch.add(x_main, x_branch)

        # Final layers
        x_combined = self.flatten(x_combined)
        x_out = self.fc(x_combined)

        return x_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # MNIST image shape: (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor)