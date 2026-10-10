import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the three convolutional groups
        self.conv_groups = nn.ModuleList()

        # Group 1: 1x1 convolution
        self.conv_groups.append(nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        ))

        # Group 2: 3x3 convolution
        self.conv_groups.append(nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        ))

        # Group 3: 5x5 convolution
        self.conv_groups.append(nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        ))

        # Concatenate the outputs from the three groups
        self.concat = nn.Sequential()

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features='mask', out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process each group separately
        group_outputs = []  # output tensor shape: (unknown)
        for group in self.conv_groups:  # output tensor shape: (unknown)
            group_outputs.append(group(x))

        # Concatenate along the channel dimension
        concatenated = torch.cat(group_outputs, dim=1)  # output tensor shape: (1, 192, 32, 32)

        # Pass through fully connected layers
        output = self.fc_layers(concatenated)
        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 images are 32x32
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)