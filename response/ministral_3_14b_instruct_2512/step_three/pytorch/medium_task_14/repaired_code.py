import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Sequential blocks with conv, batch norm, and ReLU
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )

        # Parallel branch
        self.parallel_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # Average pooling for downsampling
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=256*4*4, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():  # output tensor shape: (unknown)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process through sequential blocks
        block1_out = self.block1(x)  # output tensor shape: (1, 64, 32, 32)
        block2_out = self.block2(block1_out)  # output tensor shape: (1, 128, 32, 32)
        block3_out = self.block3(block2_out)  # output tensor shape: (1, 256, 32, 32)

        # Process through parallel branch
        parallel_out = self.parallel_conv(x)  # output tensor shape of this layer should be (1, 256, 32, 32)

        # Add outputs from all paths
        combined = block3_out + parallel_out

        # Apply average pooling
        pooled = self.avg_pool(combined)

        # Flatten for fully connected layers
        flattened = pooled.view(pooled.size(0), -1)

        # Pass through fully connected layers
        fc1_out = self.fc1(flattened)
        fc1_out = F.relu(fc1_out)
        output = self.fc2(fc1_out)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)