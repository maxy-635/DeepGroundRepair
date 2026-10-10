import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Feature extraction path 1: 1x1 convolution
        self.conv1x1_path = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Feature extraction path 2: sequence of convolutions: 1x1, 1x7, 7x1
        self.conv_sequence_path = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.ReLU(inplace=True)
        )

        # Concatenate outputs of the two paths
        self.concat = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Branch directly to the input
        self.branch = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # Path 1 output
        path1_output = self.conv1x1_path(x)

        # Path 2 output
        path2_output = self.conv_sequence_path(x)

        # Concatenate outputs
        concatenated_output = torch.cat((path1_output, path2_output), dim=1)
        aligned_output = self.concat(concatenated_output)

        # Branch output
        branch_output = self.branch(x)

        # Merge outputs through addition
        merged_output = torch.add(aligned_output, branch_output)

        # Flatten and pass through fully connected layers
        flattened_output = self.fc_layers(merged_output)

        return flattened_output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)