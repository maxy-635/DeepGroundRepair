import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.pool1 = nn.MaxPool2d(kernel_size=1, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(in_features='mask', out_features='mask')
        self.reshape1 = nn.Linear(128, 4 * 4 * 128)


        # Block 2
        self.conv1 = nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.flatten2 = nn.Flatten()
        self.fc2 = nn.Linear(128, 10)

    def calculate_output_size(self, in_channels, in_height, in_width, in_depth):
        # Calculate the output size of the first block
        output_size = (in_height // self.pool1.kernel_size[0]) * self.pool1.stride[0] + self.pool1.padding[0]
        output_size = (output_size // self.pool2.kernel_size[0]) * self.pool2.stride[0] + self.pool2.padding[0]
        output_size = (output_size // self.pool3.kernel_size[0]) * self.pool3.stride[0] + self.pool3.padding[0]
        return output_size

    def forward(self, x):
        # Block 1
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.flatten1(x)
        x = self.fc1(x)
        x = self.reshape1(x)
        x = x.view(x.size(0), 4, 4, 128)

        # Block 2
        x = torch.cat([
            self.conv1(x),
            self.conv2(x),
            self.conv3(x),
            self.pool4(x)
        ], dim=1)
        x = self.flatten2(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)