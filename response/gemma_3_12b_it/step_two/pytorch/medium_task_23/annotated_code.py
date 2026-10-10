import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: Single 1x1 convolution
        self.path1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Path 2: 1x1 followed by 1x7 and 7x1 convolutions
        self.path2_conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path2_conv2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path2_conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3)

        # Path 3: 1x1 followed by two sets of 1x7 and 7x1 convolutions
        self.path3_conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path3_conv2_1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path3_conv2_1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.path3_conv2_2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path3_conv2_2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3)

        # Path 4: Average pooling followed by 1x1 convolution
        self.path4_pool = nn.functional.avg_pool2d(input='mask', kernel_size='mask', stride='mask', padding='mask')
        self.path4_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Fully connected layer
        self.fc = nn.Linear(in_features=64 * 32 * 32, out_features=10) # Assuming input size of 32x32

    def forward(self, x):
        # Path 1
        path1_out = self.path1(x)

        # Path 2
        path2_conv1 = self.path2_conv1(x)
        path2_conv2_1 = self.path2_conv2_1(path2_conv1)
        path2_conv2_2 = self.path2_conv2_2(path2_conv2_1)
        path2_out = path2_conv2_2

        # Path 3
        path3_conv1 = self.path3_conv1(x)
        path3_conv2_1_1 = self.path3_conv2_1_1(path3_conv1)
        path3_conv2_1_2 = self.path3_conv2_1_2(path3_conv2_1_1)
        path3_conv2_2_1 = self.path3_conv2_2_1(path3_conv2_1_2)
        path3_conv2_2_2 = self.path3_conv2_2_2(path3_conv2_2_1)
        path3_out = path3_conv2_2_2

        # Path 4
        path4_pool = self.path4_pool(x)
        path4_conv = self.path4_conv(path4_pool)
        path4_out = path4_conv

        # Concatenate paths
        combined_out = torch.cat((path1_out, path2_out, path3_out, path4_out), dim=1)

        # Flatten
        flattened_out = torch.nn.Flatten()(combined_out)

        # Fully connected layer
        output = self.fc(flattened_out)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)