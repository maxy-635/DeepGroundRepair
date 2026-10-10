import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2)
        )
        self.branch_path = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)
        )
        self.fuse = nn.Sequential(
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        main_path_output = self.main_path(x)  # output tensor shape of this layer should be (1, 16, 32, 32)
        branch_path_output = self.branch_path(x)  # output tensor shape: (1, 16, 32, 32)
        
        # Assuming input size is 32x32, and each main path conv layer outputs 16 channels.
        # We concatenate the outputs of the three convolutional layers in the main path.
        main_path_output = torch.cat([main_path_output, main_path_output, main_path_output], dim=1)
        
        # Add the outputs from both paths
        fused_output = torch.add(main_path_output, branch_path_output)
        
        # Pass the combined output through the fully connected layers
        output = self.fuse(fused_output)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)