import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolutional layer
        self.initial_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Block 1 components
        self.conv1x1_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.depthwise_conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32)
        self.conv1x1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Block 2 components
        self.fc = nn.Linear(in_features=32*28*28, out_features=10)  # Assuming MNIST images are 28x28

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Initial convolution
        x = self.initial_conv(x)  # output tensor shape: (1, 32, 28, 28)

        # Block 1
        # Split the input into two groups along the last dimension
        x_split = torch.split(x, x.size(-1)//2, dim=-1)  # output tensor shape: (unknown)
        x_group1 = x_split[0]  # output tensor shape: (1, 32, 28, 14)
        x_group2 = x_split[1]  # output tensor shape: (1, 32, 28, 14)

        # Process the first group
        x_group1 = self.conv1x1_1(x_group1)  # output tensor shape: (1, 32, 28, 14)
        x_group1 = self.depthwise_conv(x_group1)  # output tensor shape: (1, 32, 28, 14)
        x_group1 = self.conv1x1_2(x_group1)  # output tensor shape: (1, 32, 28, 14)

        # Concatenate the processed group with the unmodified group
        x = torch.cat((x_group1, x_group2), dim=-1)  # output tensor shape: (1, 32, 28, 28)

        # Block 2: Channel shuffling
        # Get the shape of the input
        _, _, height, width = x.size()  # output tensor shape: (1); (1); (1)
        channels = x.size(-1)  # output tensor shape: (1)

        # Reshape into four groups
        x = torch.reshape(x, (x.size(0), 4, height, width, channels//4))
        x = torch.permute(x, (0, 3, 1, 2, 4))
        x = torch.reshape(x, (x.size(0), height, width, channels))

        # Flatten and pass through fully connected layer
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor)