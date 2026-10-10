import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1
        self.path1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size='mask', stride='mask', padding='mask')
        self.relu1 = nn.ReLU(inplace=True)

        # Path 2
        self.path2_conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.path2_conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 7), stride=1, padding=0)
        self.path2_conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(7, 1), stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)

        # Path 3
        self.path3_conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.path3_conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 7), stride=1, padding=0)
        self.path3_conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(7, 1), stride=1, padding=0)
        self.path3_conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 7), stride=1, padding=0)
        self.path3_conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(7, 1), stride=1, padding=0)
        self.relu3 = nn.ReLU(inplace=True)

        # Path 4
        self.path4_pool = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        self.path4_conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.relu4 = nn.ReLU(inplace=True)

        # Concatenation
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=16 * 32 * 32 * 4, out_features=10) # Adjusted input size for concatenation

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Path 1
        path1 = self.relu1(self.path1(x))  # output tensor shape of this layer should be (1, 16, 26, 26)

        # Path 2
        path2 = self.relu2(self.path2_conv3(self.relu2(self.path2_conv2(self.path2_conv1(x)))))  # output tensor shape: (1, 16, 26, 26)

        # Path 3
        path3 = self.relu3(self.path3_conv5(self.relu3(self.path3_conv4(self.relu3(self.path3_conv3(self.relu3(self.path3_conv2(self.path3_conv1(x)))))))))  # output tensor shape: (1, 16, 20, 20)

        # Path 4
        path4 = self.relu4(self.path4_conv(self.path4_pool(x)))  # output tensor shape: (1, 16, 4, 4)

        # Concatenation
        x = torch.cat((path1, path2, path3, path4), dim=1)

        # Flatten and Fully Connected Layer
        x = self.flatten(x)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])