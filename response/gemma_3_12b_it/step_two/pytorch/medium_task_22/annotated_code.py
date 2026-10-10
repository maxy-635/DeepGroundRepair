import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 3x3 Convolutions
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Branch 2: 1x1 + 2x 3x3 Convolutions
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Branch 3: Max Pooling
        self.branch3 = nn.MaxPool2d(kernel_size='mask', stride='mask', padding='mask')

        # Feature Fusion
        # Assuming input size to branches is the same, otherwise, adjust dimensions accordingly.
        # We'll assume the input is 32x32 after the convolutional layers

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Pass input through each branch
        branch1_out = self.branch1(x)  # output tensor shape: (1, 64, 32, 32)
        branch2_out = self.branch2(x)  # output tensor shape: (1, 64, 32, 32)
        branch3_out = self.branch3(x)  # output tensor shape of this layer should be (1, 3, 32, 32)

        # Concatenate the outputs of the branches
        fused_features = torch.cat((branch1_out, branch2_out, branch3_out), dim=1)

        # Flatten the fused features
        flattened_features = torch.flatten(fused_features, 1)

        # Pass through fully connected layers
        x = torch.relu(self.fc1(flattened_features))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)