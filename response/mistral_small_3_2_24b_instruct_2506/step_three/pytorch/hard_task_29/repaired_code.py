import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Main path and branch path
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        # Second block: Three max pooling layers with different scales
        self.pool1 = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=1029, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # First block: Main path and branch path
        main_path = self.conv1(x)  # output tensor shape: (1, 32, 28, 28)
        main_path = self.conv2(main_path)  # output tensor shape: (1, 1, 28, 28)
        branch_path = x  # output tensor shape: (1, 1, 28, 28)
        block1_output = main_path + branch_path  # output tensor shape: (1, 1, 28, 28)

        # Second block: Three max pooling layers
        pool1_output = self.pool1(block1_output)  # output tensor shape: (1, 1, 28, 28)
        pool2_output = self.pool2(block1_output)  # output tensor shape: (1, 1, 14, 14)
        pool3_output = self.pool3(block1_output)  # output tensor shape: (1, 1, 7, 7)

        # Flatten and concatenate
        pool1_output = torch.flatten(pool1_output, start_dim=1)  # output tensor shape: (1, 784)
        pool2_output = torch.flatten(pool2_output, start_dim=1)  # output tensor shape: (1, 196)
        pool3_output = torch.flatten(pool3_output, start_dim=1)  # output tensor shape: (1, 49)
        block2_output = torch.cat((pool1_output, pool2_output, pool3_output), dim=1)  # output tensor shape: (1, 1029)

        # Fully connected layers
        x = self.fc1(block2_output)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])