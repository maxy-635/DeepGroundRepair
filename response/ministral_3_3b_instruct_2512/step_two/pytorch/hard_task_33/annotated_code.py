import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the block
        self.block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Define three branches
        self.branch1 = nn.Sequential(self.block)
        self.branch2 = nn.Sequential(self.block)
        self.branch3 = nn.Sequential(self.block)

        # Concatenate the outputs of the three branches
        self.concat = nn.Sequential(
            nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask'),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Pass the input through the three branches
        branch1_output = self.branch1(x)  # output tensor shape: (1, 16, 28, 28)
        branch2_output = self.branch2(x)  # output tensor shape: (1, 16, 28, 28)
        branch3_output = self.branch3(x)  # output tensor shape: (1, 16, 28, 28)

        # Concatenate the outputs of the three branches
        concatenated_output = torch.cat((branch1_output, branch2_output, branch3_output), dim=1)  # output tensor shape: (1, 48, 28, 28)

        # Pass the concatenated output through the flattening and fully connected layer
        output = self.concat(concatenated_output)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (batch_size, 10)