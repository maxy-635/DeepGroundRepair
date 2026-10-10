import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First path: 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Second path: 1x1 convolution followed by two 3x3 convolutions
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

        # Third path: 1x1 convolution followed by a single 3x3 convolution
        self.path3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

        # Fourth path: max pooling followed by 1x1 convolution
        self.path4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Flatten and dense layers
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features=262144, out_features=128)
        self.output = nn.Linear(in_features=128, out_features=10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process each path
        out1 = self.path1(x)  # output tensor shape: (1, 64, 32, 32)
        out2 = self.path2(x)  # output tensor shape: (1, 64, 32, 32)
        out3 = self.path3(x)  # output tensor shape: (1, 64, 32, 32)
        out4 = self.path4(x)  # output tensor shape: (1, 64, 32, 32)

        # Concatenate the outputs
        out = torch.cat((out1, out2, out3, out4), dim=1)  # output tensor shape: (1, 256, 32, 32)

        # Flatten and pass through dense layers
        out = self.flatten(out)  # output tensor shape: (1, 262144)
        out = self.dense(out)
        out = self.output(out)
        out = self.softmax(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])