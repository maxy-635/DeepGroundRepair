import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Split into three groups with depthwise separable convolutions
        self.group1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0, groups=3),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.group2 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, groups=3),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Concatenate the outputs from the three groups
        self.concat1 = nn.Sequential(nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0), nn.ReLU())




        # Second block: Multiple branches for feature extraction
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Concatenate the outputs from all branches
        self.concat2 = nn.Sequential(
            nn.Conv2d(12, 3, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 4 * 4, 10)  # Assuming input size is 32x32 after pooling

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First block
        group1_out = self.group1(x)  # output tensor shape: (1, 3, 32, 32)
        group2_out = self.group2(x)  # output tensor shape: (1, 3, 32, 32)
        group3_out = self.group3(x)  # output tensor shape: (1, 3, 32, 32)

        # Concatenate along channel dimension
        concat1_out = torch.cat([group1_out, group2_out, group3_out], dim=1)  # output tensor shape: (1, 9, 32, 32)
        concat1_out = self.concat1(conv2d(concat1_out))

        # Second block
        branch1_out = self.branch1(concat1_out)
        branch2_out = self.branch2(concat1_out)
        branch3_out = self.branch3(concat1_out)
        branch4_out = self.branch4(concat1_out)

        # Concatenate along channel dimension
        concat2_out = torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)
        concat2_out = self.concat2(concat2_out)

        # Final layers
        x = self.flatten(concat2_out)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)