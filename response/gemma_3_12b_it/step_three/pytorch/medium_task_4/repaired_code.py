import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: Two blocks of convolution + average pooling
        self.path1_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.path1_pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.path1_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.path1_pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # Path 2: Single convolutional layer
        self.path2_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Flatten and fully connected layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64 * 8 * 8, out_features=10)  # Assuming input size is 32x32, after two pooling layers it becomes 8x8

    def forward(self, x):
        # Path 1
        x_path1 = self.path1_conv1(x)
        x_path1 = self.path1_pool1(x_path1)
        x_path1 = self.path1_conv2(x_path1)
        x_path1 = self.path1_pool2(x_path1)

        # Path 2
        x_path2 = self.path2_conv(x)

        # Combine paths
        x_combined = x_path1 + x_path2

        # Flatten and fully connected layer
        x_flattened = self.flatten(x_combined)
        x_output = self.fc(x_flattened)

        return x_output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)