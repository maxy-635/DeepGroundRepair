import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three parallel paths with different max pooling scales
        self.maxpool1x1 = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.maxpool2x2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.maxpool4x4 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)

        self.flatten = nn.Flatten()

        # Fully connected layer and reshaping
        self.fc1 = nn.Linear(in_features='mask', out_features=128)
        self.reshape = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=4*4*16)  # Reshape to 4D tensor
        )

        # Block 2: Four parallel paths with different convolution and pooling strategies
        # Path 1: Single 1x1 convolution
        self.conv1x1_path1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)

        # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
        self.conv1x1_path2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv1x7_path2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 7), stride=1, padding=0)
        self.conv7x1_path2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(7, 1), stride=1, padding=0)

        # Path 3: 1x1 convolution followed by alternating 7x1 and 1x7 convolutions
        self.conv1x1_path3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv7x1_path3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(7, 1), stride=1, padding=0)
        self.conv1x7_path3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 7), stride=1, padding=0)

        # Path 4: Average pooling with 1x1 convolution
        self.avgpool_path4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1x1_path4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)

        # Final fully connected layers
        self.fc2 = nn.Linear(in_features=64*64, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)  # Assuming 10 classes for MNIST

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Block 1: Three parallel paths
        x1 = self.maxpool1x1(x)  # output tensor shape: (1, 1, 28, 28)
        x1 = self.flatten(x1)  # output tensor shape: (1, 784)
        x1 = self.dropout1(x1)  # output tensor shape: (1, 784)

        x2 = self.maxpool2x2(x)  # output tensor shape: (1, 1, 14, 14)
        x2 = self.flatten(x2)  # output tensor shape: (1, 196)
        x2 = self.dropout2(x2)  # output tensor shape: (1, 196)

        x3 = self.maxpool4x4(x)  # output tensor shape: (1, 1, 7, 7)
        x3 = self.flatten(x3)  # output tensor shape: (1, 49)
        x3 = self.dropout3(x3)  # output tensor shape: (1, 49)

        # Concatenate the outputs
        x = torch.cat((x1, x2, x3), dim=1)  # output tensor shape: (1, 1029)

        # Fully connected layer and reshaping
        x = self.fc1(x)
        x = self.reshape(x)

        # Block 2: Four parallel paths
        # Path 1
        path1 = self.conv1x1_path1(x)

        # Path 2
        path2 = self.conv1x1_path2(x)
        path2 = self.conv1x7_path2(path2)
        path2 = self.conv7x1_path2(path2)

        # Path 3
        path3 = self.conv1x1_path3(x)
        path3 = self.conv7x1_path3(path3)
        path3 = self.conv1x7_path3(path3)

        # Path 4
        path4 = self.avgpool_path4(x)
        path4 = self.conv1x1_path4(path4)

        # Concatenate the outputs
        x = torch.cat((path1, path2, path3, path4), dim=1)

        # Flatten and final fully connected layers
        x = self.flatten(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # MNIST input shape: (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)