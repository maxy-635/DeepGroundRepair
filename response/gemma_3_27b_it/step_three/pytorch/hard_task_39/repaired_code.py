import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Max Pooling and Concatenation
        self.maxpool1 = nn.MaxPool2d(kernel_size=1, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.flatten = nn.Flatten()

        # Fully connected layer and reshape for Block 2
        self.fc1 = nn.Linear(in_features=1029, out_features=64)
        

        # Block 2: Multi-Branch Feature Extraction
        self.conv1x1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1)
        self.conv3x3 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.maxpool_3x3 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)

        # Classification
        self.flatten_final = nn.Flatten()
        self.fc2 = nn.Linear(in_features=32 * 9, out_features=10) # Adjusted input features based on concatenation

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Block 1
        pool1 = self.maxpool1(x)  # output tensor shape: (1, 1, 28, 28)
        pool2 = self.maxpool2(x)  # output tensor shape: (1, 1, 14, 14)
        pool3 = self.maxpool3(x)  # output tensor shape: (1, 1, 7, 7)
        
        flat_pool1 = self.flatten(pool1)  # output tensor shape: (1, 784)
        flat_pool2 = self.flatten(pool2)  # output tensor shape: (1, 196)
        flat_pool3 = self.flatten(pool3)  # output tensor shape: (1, 49)
        
        block1_output = torch.cat((flat_pool1, flat_pool2, flat_pool3), dim=1)  # output tensor shape: (1, 1029)

        # Intermediate Layer
        block1_output = self.fc1(block1_output)
        block1_output = torch.reshape(block1_output, (block1_output.shape[0], 1, 9, 9))

        # Block 2
        branch1 = self.conv1x1(block1_output)
        branch2 = self.conv3x3(block1_output)
        branch3 = self.conv5x5(block1_output)
        branch4 = self.maxpool_3x3(block1_output)

        block2_output = torch.cat((branch1, branch2, branch3, branch4), dim=1)

        # Classification
        flat_block2 = self.flatten_final(block2_output)
        output = self.fc2(flat_block2)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)