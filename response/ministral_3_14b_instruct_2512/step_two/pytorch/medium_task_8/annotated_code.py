import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path components
        self.conv1 = nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

        # Branch path component
        self.branch_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Final classification layer
        self.fc = nn.Linear(in_features=3*32*32, out_features=10)  # CIFAR-10 has 10 classes

        # Activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # the initial shape of x is: (4, 3, 32, 32)
        # Split the input into three groups along the channel dimension
        x1, x2, x3 = torch.split(x, 1, dim=1)  # output tensor shape: (4, 1, 32, 32); (4, 1, 32, 32); (4, 1, 32, 32)

        # Main path processing
        # First group remains unchanged
        # Second group goes through conv1
        x2 = self.conv1(x2)
        x2 = self.relu(x2)
        # Combine with third group
        x23 = torch.cat([x2, x3], dim=1)
        # Additional conv2
        x23 = self.conv2(x23)
        x23 = self.relu(x23)

        # Concatenate all three groups
        main_output = torch.cat([x1, x23], dim=1)

        # Branch path processing
        branch_output = self.branch_conv(x)
        branch_output = self.relu(branch_output)

        # Combine main and branch outputs
        combined = torch.add(main_output, branch_output)

        # Flatten and classify
        combined = combined.view(combined.size(0), -1)
        output = self.fc(combined)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(4, 3, 32, 32)  # batch_size=4
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Should be [4, 10]