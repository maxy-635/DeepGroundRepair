import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # Fully connected layer and reshaping between Block 1 and Block 2
        self.fc1 = nn.Linear(in_features=1029, out_features=4 * 7 * 7)
        self.reshape = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*7*7, 4*7*7)  # Identity transformation for reshaping
        )

        # Block 2: Four parallel paths with different convolution and pooling strategies
        # Path 1: Single 1x1 convolution
        self.path1_conv1x1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Path 2: 1x1 conv followed by 1x7 and 7x1 convolutions
        self.path2_conv1x1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path2_conv1x7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.path2_conv7x1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0))

        # Path 3: 1x1 conv followed by two alternating 7x1 and 1x7 convolutions
        self.path3_conv1x1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path3_conv7x1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.path3_conv1x7_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.path3_conv7x1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0))

        # Path 4: Average pooling followed by 1x1 convolution
        self.path4_avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.path4_conv1x1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Concatenate all paths
        self.concat = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Final classification layers
        self.fc2 = nn.Linear(in_features=256*7*7, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Block 1: Three parallel paths with different max pooling scales
        x1 = self.maxpool1x1(x)  # output tensor shape: (1, 1, 28, 28)
        x1 = self.flatten(x1)  # output tensor shape: (1, 784)
        x1 = self.dropout1(x1)  # output tensor shape: (1, 784)

        x2 = self.maxpool2x2(x)  # output tensor shape: (1, 1, 14, 14)
        x2 = self.flatten(x2)  # output tensor shape: (1, 196)
        x2 = self.dropout2(x2)  # output tensor shape: (1, 196)

        x3 = self.maxpool4x4(x)  # output tensor shape: (1, 1, 7, 7)
        x3 = self.flatten(x3)  # output tensor shape: (1, 49)
        x3 = self.dropout3(x3)  # output tensor shape: (1, 49)

        # Concatenate the outputs of Block 1
        x_concat = torch.cat((x1, x2, x3), dim=1)  # output tensor shape: (1, 1029)

        # Reshape and fully connected layer between Block 1 and Block 2
        x_concat = self.fc1(x_concat)
        x_concat = self.reshape(x_concat)
        x_concat = x_concat.view(-1, 1, 7, 7)

        # Block 2: Four parallel paths with different convolution and pooling strategies
        # Path 1: Single 1x1 convolution
        path1 = self.path1_conv1x1(x_concat)

        # Path 2: 1x1 conv followed by 1x7 and 7x1 convolutions
        path2 = self.path2_conv1x1(x_concat)
        path2 = self.path2_conv1x7(path2)
        path2 = self.path2_conv7x1(path2)

        # Path 3: 1x1 conv followed by two alternating 7x1 and 1x7 convolutions
        path3 = self.path3_conv1x1(x_concat)
        path3 = self.path3_conv7x1_1(path3)
        path3 = self.path3_conv1x7_1(path3)
        path3 = self.path3_conv7x1_2(path3)

        # Path 4: Average pooling followed by 1x1 convolution
        path4 = self.path4_avgpool(x_concat)
        path4 = self.path4_conv1x1(path4)

        # Concatenate all paths
        x_concat = torch.cat((path1, path2, path3, path4), dim=1)
        x_concat = self.concat(x_concat)

        # Final classification layers
        x_concat = x_concat.view(x_concat.size(0), -1)
        x_concat = self.fc2(x_concat)
        x_concat = F.relu(x_concat)
        x_concat = self.fc3(x_concat)

        return x_concat

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)