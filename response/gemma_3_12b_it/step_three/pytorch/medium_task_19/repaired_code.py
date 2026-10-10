import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 convolution for dimensionality reduction
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Branch 2: 1x1 conv followed by 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

        # Branch 3: 1x1 conv followed by 5x5 conv
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        )

        # Branch 4: 3x3 max pooling followed by 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)  # Assuming input size after branches is 32x32 -> 8x8 after pooling and convolutions
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Pass through the branches
        branch1_out = self.branch1(x)  # output tensor shape: (1, 64, 32, 32)
        branch2_out = self.branch2(x)  # output tensor shape: (1, 64, 32, 32)
        branch3_out = self.branch3(x)  # output tensor shape: (1, 64, 32, 32)
        branch4_out = self.branch4(x)  # output tensor shape of this layer should be (1, 64, 32, 32)

        # Concatenate the outputs
        concatenated = torch.cat((branch1_out, branch2_out, branch3_out, branch4_out), dim=1)

        # Flatten the features
        flattened = torch.nn.Flatten()(concatenated)

        # Pass through the fully connected layers
        fc1_out = torch.relu(self.fc1(flattened))
        output = self.fc2(fc1_out)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (CIFAR-10 images are 32x32 with 3 channels)
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)