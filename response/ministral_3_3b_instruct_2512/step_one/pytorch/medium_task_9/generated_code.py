import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolutional layer to reduce dimensionality to 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Basic block definition
        self.basic_block = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Branch convolutional layer for feature extraction
        self.branch_conv = nn.Conv2d(3, 16, kernel_size=1, stride=1)

        # Second basic block
        self.basic_block2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Branch convolutional layer for second basic block
        self.branch_conv2 = nn.Conv2d(16, 16, kernel_size=1, stride=1)

        # Adaptive average pooling and fully connected layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        # First basic block path
        x_main = self.conv1(x)
        x_main = self.basic_block(x_main)

        # Branch path
        x_branch = self.branch_conv(x)

        # Feature fusion by addition
        x_fused = torch.add(x_main, x_branch)

        # Second basic block path
        x_main = self.basic_block2(x_fused)

        # Second branch path
        x_branch = self.branch_conv2(x_fused)

        # Feature fusion by addition again
        x_fused = torch.add(x_main, x_branch)

        # Downsampling and classification
        x_pool = self.pool(x_fused)
        x_flat = self.flatten(x_pool)
        output = self.fc(x_flat)

        return output

    def flatten(self, x):
        return x.view(x.size(0), -1)

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor for CIFAR-10 (batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (1, 10)