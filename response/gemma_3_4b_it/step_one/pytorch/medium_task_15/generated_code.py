import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)

        # Batch Normalization and ReLU
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 1 * 1, 10)  # Adjust dimensions as needed
        self.fc2 = nn.Linear(10, 10)

        # Reshape and concatenate (simplified for demonstration)
        self.reshape = nn.Flatten()
        self.concat = nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0) # Dummy concat layer


        # Dimensionality reduction and downsampling
        self.conv_reduce = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0)
        self.avgpool2 = nn.AvgPool2d(kernel_size=8, stride=8) # Downsample to 1x1

        # Final fully connected layer
        self.fc_out = nn.Linear(16, 10)

    def forward(self, x):
        # Convolution, Batch Norm, ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Global Average Pooling
        x = self.avgpool(x)

        # Fully connected layers
        x = self.reshape(x)
        x = self.fc1(x)
        x = self.fc2(x)

        # Reshape and concatenate (simplified)
        x = self.concat(x)

        # Dimensionality reduction and downsampling
        x = self.conv_reduce(x)
        x = self.avgpool2(x)

        # Final fully connected layer
        x = self.fc_out(x)

        return x


if __name__ == "__main__":
    # Dummy input tensor (Batch size, Channels, Height, Width)
    input_tensor = torch.randn(1, 3, 32, 32)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)