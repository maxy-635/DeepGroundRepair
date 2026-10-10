import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Feature extraction branches
        branch1 = nn.Sequential(  # output tensor shape: (unknown)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        branch2 = nn.Sequential(  # output tensor shape: (unknown)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

        branch3 = nn.Sequential(  # output tensor shape: (unknown)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

        # Concatenation and dimension adjustment
        self.concat = nn.Sequential(
            nn.Conv2d(in_channels=64 * 3, out_channels=3, kernel_size=1, stride=1, padding=0)
        )

        # Residual connection
        self.residual_connection = lambda x: x

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(in_features=3 * 32 * 32, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Feature extraction
        branch1_out = branch1(x)
        branch2_out = branch2(x)
        branch3_out = branch3(x)

        # Concatenation
        concat_out = torch.cat((branch1_out, branch2_out, branch3_out), dim=1)

        # Dimension adjustment
        concat_out = self.concat(concat_out)

        # Residual connection
        x = torch.add(x, concat_out)

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)