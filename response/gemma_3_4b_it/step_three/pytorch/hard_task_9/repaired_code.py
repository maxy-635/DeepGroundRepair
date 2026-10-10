import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Branch 3: 1x1 convolution followed by two 3x3 convolutions
        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Concatenate the outputs of the three branches
        self.concat_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Adjust the input size based on the expected output from the convolutional branches
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        main_path = x  # output tensor shape: (1, 3, 32, 32)

        # Branch 1
        branch1 = self.branch1(x)  # output tensor shape: (1, 64, 32, 32)

        # Branch 2
        branch2 = self.branch2(x)  # output tensor shape: (1, 64, 32, 32)

        # Branch 3
        branch3 = self.branch3(x)  # output tensor shape: (1, 64, 32, 32)

        # Concatenate the branches
        concat = self.concat_layer(torch.cat([main_path, branch1, branch2, branch3], dim=1))

        # Fully connected layers
        concat = concat.view(concat.size(0), -1)
        output = self.fc1(concat)
        output = torch.nn.ReLU(inplace=True)(output)
        output = self.fc2(output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)