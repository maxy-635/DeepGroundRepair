import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path: 7x7 depthwise convolution + 2x 1x1 pointwise convolutions
        self.main_path = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(66),  # Changed to match output channels of previous layer
            nn.ReLU(inplace=True)
        )
        # Branch path: direct connection to input
        self.branch_path = nn.Conv2d(3, 64, kernel_size=1, stride=1)
        # Addition operation
        self.add(main_path_output, branch_path_output)
        # Flattening and fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 128)  # Assuming 32x32 input after convolution
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Process main path
        main_path_output = self.main_path(x)
        # Process branch path
        branch_path_output = self.branch_path(x)
        # Combine outputs
        combined_output = self.add(main_path_output, branch_path_output)
        # Flatten and process through fully connected layers
        flattened_output = self.flatten(combined_output)
        fc1_output = self.fc1(flattened_output)
        fc2_output = self.fc2(nn.ReLU()(fc1_output))
        return fc2_output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 image dimensions
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
    print("Output:", output_tensor)