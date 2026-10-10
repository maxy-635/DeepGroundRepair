import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.main_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.main_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.main_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Branch path
        self.branch_conv = nn.Conv2d(in_channels=3, out_channels='mask', kernel_size=3, stride=1, padding=1)
        self.branch_pool = nn.MaxPool2d(kernel_size='mask', stride='mask', padding='mask')

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64*6*6 + 16*12*12, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        main_out = self.main_conv1(x)  # output tensor shape: (1, 32, 32, 32)
        main_out = self.main_pool1(main_out)  # output tensor shape: (1, 32, 16, 16)
        main_out = self.main_conv2(main_out)  # output tensor shape: (1, 64, 16, 16)
        main_out = self.main_pool2(main_out)  # output tensor shape: (1, 64, 8, 8)

        # Branch path
        branch_out = self.branch_conv(x)  # output tensor shape of this layer should be (1, 64, 32, 32)
        branch_out = self.branch_pool(branch_out)  # output tensor shape of this layer should be (1, 16, 32, 32)

        # Combine outputs
        combined_out = torch.add(main_out, branch_out)

        # Flatten and fully connected layers
        combined_out = self.flatten(combined_out)
        combined_out = self.fc1(combined_out)
        combined_out = self.fc2(combined_out)

        return combined_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)