import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)
        )

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )

        # Branch 3: 1x1 convolution followed by two 3x3 convolutions
        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )

        # Concatenate the outputs of the three branches
        self.concat = nn.Conv2d(192, 3, kernel_size=1, stride=1, padding=0)

        # Direct connection to the input
        self.input_to_output = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)
        )

        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        # Extract features from each branch
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)

        # Concatenate the outputs of the three branches
        concat_out = torch.cat((branch1_out, branch2_out, branch3_out), dim=1)

        # Adjust dimensions using 1x1 convolution
        adjusted_out = self.concat(concat_out)

        # Direct connection output
        direct_out = self.input_to_output(x)

        # Add the outputs of the main path and the branch
        fused_out = adjusted_out + direct_out

        # Classification through fully connected layers
        pooled = self.fc_layers(fused_out)

        return pooled

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor with batch size 1, 3 channels, 32x32 spatial dimensions
    output_tensor = model(input_tensor)
    print(output_tensor.shape)