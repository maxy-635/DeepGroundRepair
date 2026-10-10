import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Splitting and feature extraction
        self.block1_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block1_conv3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.block1_conv5 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.block1_dropout = nn.Dropout(p=0.2)
        self.block1_concat = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Block 2: Four branches for feature fusion
        self.branch1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Concatenation and final layers
        self.concat = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 8 * 8, 10)

    def forward(self, x):
        # Block 1: Splitting and feature extraction
        conv1_out = self.block1_conv1(x)
        conv3_out = self.block1_conv3(x)
        conv5_out = self.block1_conv5(x)

        # Concatenate outputs from the three groups
        concat_out = torch.cat((conv1_out, conv3_out, conv5_out), dim=1)
        concat_out = self.block1_dropout(concat_out)
        concat_out = self.block1_concat(concat_out)

        # Block 2: Four branches for feature fusion
        branch1_out = self.branch1(concat_out)
        branch2_out = self.branch2(concat_out)
        branch3_out = self.branch3(concat_out)
        branch4_out = self.branch4(concat_out)

        # Concatenate outputs from all branches
        branch_out = torch.cat((branch1_out, branch2_out, branch3_out, branch4_out), dim=1)
        branch_out = self.concat(branch_out)

        # Final layers
        x = self.flatten(branch_out)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (batch_size, 10)