import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.pool1_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv1_2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv1_3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu1_3 = nn.ReLU(inplace=True)
        self.pool1_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Parallel branch
        self.conv_parallel = nn.Conv2d(in_channels=1, out_channels='mask', kernel_size='mask', stride='mask', padding='mask')
        self.relu_parallel = nn.ReLU(inplace=True)

        # Block 2
        self.conv2_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.pool2_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2_2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2_3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu2_3 = nn.ReLU(inplace=True)
        self.pool2_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Flatten and Linear
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(256 * 7 * 7, 10)  # Assuming 10 classes for MNIST

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Branch 1
        branch1 = x  # output tensor shape: (1, 1, 28, 28)
        branch1 = self.relu1_1(self.conv1_1(branch1))  # output tensor shape: (1, 32, 28, 28)
        branch1 = self.pool1_1(branch1)  # output tensor shape: (1, 32, 14, 14)
        branch1 = self.relu1_2(self.conv1_2(branch1))  # output tensor shape: (1, 64, 14, 14)
        branch1 = self.pool1_2(branch1)  # output tensor shape: (1, 64, 7, 7)
        branch1 = self.relu1_3(self.conv1_3(branch1))  # output tensor shape: (1, 128, 7, 7)
        branch1 = self.pool1_3(branch1)  # output tensor shape: (1, 128, 3, 3)

        # Branch 2
        branch2 = x  # output tensor shape: (1, 1, 28, 28)
        branch2 = self.relu_parallel(self.conv_parallel(branch2))  # output tensor shape of this layer should be (1, 128, 3, 3)

        # Combine branches
        combined = torch.add(branch1, branch2)

        # Concatenate
        concatenated = torch.cat((branch1, branch2), dim=1)

        # Flatten and Linear
        flattened = self.flatten(concatenated)
        output = self.linear(flattened)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)