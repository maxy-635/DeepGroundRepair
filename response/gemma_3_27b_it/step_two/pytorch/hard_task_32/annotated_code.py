import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the specialized block
        def specialized_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
                nn.Dropout(p=0.2),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
                nn.Dropout(p=0.2)
            )

        # Create three branches
        self.branch1 = specialized_block(1, 16)
        self.branch2 = specialized_block(1, 32)
        self.branch3 = specialized_block(1, 64)

        # Concatenate the outputs from the branches
        self.concat = nn.Conv2d(in_channels=16 + 32 + 64, out_channels=128, kernel_size=1, stride=1, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Pass input through each branch
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)

        # Concatenate the outputs
        concat_out = torch.cat([out1, out2, out3], dim=1)

        # Pass through a 1x1 convolution to reduce dimensionality
        concat_out = self.concat(concat_out)

        # Flatten the tensor for fully connected layers
        concat_out = concat_out.view(-1, 128 * 7 * 7)

        # Pass through fully connected layers
        out = F.relu(self.fc1(concat_out))
        out = F.softmax(self.fc2(out), dim=1)

        return out


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)