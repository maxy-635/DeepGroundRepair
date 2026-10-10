import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 convolution, 3x3 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        # Branch 3: Max pooling
        self.branch3 = nn.MaxPool2d(kernel_size='mask', stride='mask', padding='mask')

        # Concatenate the outputs from all branches
        self.concat_layer = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Adjust size based on input image size
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Forward pass through the branches
        branch1_output = self.branch1(x)  # output tensor shape: (1, 32, 32, 32)
        branch2_output = self.branch2(x)  # output tensor shape: (1, 32, 32, 32)
        branch3_output = self.branch3(x)  # output tensor shape of this layer should be (1, 3, 32, 32)

        # Concatenate the outputs
        concatenated_output = torch.cat((branch1_output, branch2_output, branch3_output), dim=1)

        # Flatten the concatenated output
        flattened_output = concatenated_output.view(concatenated_output.size(0), -1)

        # Fully connected layers
        fc1_output = torch.relu(self.fc1(flattened_output))
        fc2_output = torch.relu(self.fc2(fc1_output))
        fc3_output = self.fc3(fc2_output)

        return fc3_output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)