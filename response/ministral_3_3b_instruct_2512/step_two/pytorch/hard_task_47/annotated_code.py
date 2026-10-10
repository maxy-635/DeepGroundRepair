import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Splitting and concatenating outputs from three groups
        self.first_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Concatenate outputs from the three groups
        self.concat_first_block = nn.Sequential(
            nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Second block: Multiple branches for feature extraction
        self.second_block_branch1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.second_block_branch2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.second_block_branch3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )

        # Concatenate outputs from all branches
        self.concat_second_block = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 1 * 1, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First block processing
        first_block_output = self.first_block(x)  # output tensor shape: (1, 64, 32, 32)
        first_block_output = first_block_output.view(first_block_output.size(0), -1)  # output tensor shape: (1, 65536)
        first_block_output = self.concat_first_block(first_block_output)

        # Second block processing
        branch1_output = self.second_block_branch1(first_block_output)
        branch2_output = self.second_block_branch2(first_block_output)
        branch3_output = self.second_block_branch3(first_block_output)

        # Concatenate outputs from all branches
        concatenated_output = torch.cat((branch1_output, branch2_output, branch3_output), dim=1)
        concatenated_output = self.concat_second_block(concatenated_output)

        # Flatten and pass through fully connected layers
        flattened_output = concatenated_output.view(concatenated_output.size(0), -1)
        output = self.fc_layers(flattened_output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)