import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Sequential blocks with Conv2d, BatchNorm2d, and ReLU
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Parallel branch
        self.parallel_branch = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # Assuming input size after processing is 8x8
        self.fc2 = nn.Linear(512, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process through sequential blocks
        block1_out = self.block1(x)  # output tensor shape of this layer should be (1, 64, 32, 32)
        block2_out = self.block2(block1_out)  # output tensor shape: (1, 64, 32, 32)
        block3_out = self.block3(block2_out)  # output tensor shape: (1, 128, 32, 32)

        # Process through parallel branch
        parallel_out = self.parallel_branch(x)  # output tensor shape: (1, 64, 32, 32)

        # Add outputs from sequential blocks and parallel branch
        combined_output = block1_out + block2_out + block3_out + parallel_out

        # Reshape for fully connected layers
        combined_output = combined_output.view(combined_output.size(0), -1)

        # Pass through fully connected layers
        x = self.fc1(combined_output)
        x = nn.ReLU(inplace=True)(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    # Dummy input tensor for CIFAR-10 (3 channels, 32x32 pixels)
    input_tensor = torch.randn(1, 3, 32, 32)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)