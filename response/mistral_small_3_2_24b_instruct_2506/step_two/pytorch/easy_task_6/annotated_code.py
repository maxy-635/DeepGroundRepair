import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        # Branch path (identity connection)
        self.branch = nn.Sequential()

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=28*28, out_features=10)

    def forward(self, x):
        # Main path
        x_main = self.conv1(x)
        x_main = self.conv2(x_main)

        # Branch path
        x_branch = self.branch(x)

        # Combine paths
        x_combined = torch.add(x_main, x_branch)

        # Final layers
        x_flat = self.flatten(x_combined)
        x_out = self.fc(x_flat)

        return x_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)