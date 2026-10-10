import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Multi-scale average pooling
        self.avg_pool1 = nn.AvgPool2d(kernel_size=1, stride=1)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool3 = nn.AvgPool2d(kernel_size=4, stride=4)

        # Flatten and concatenate
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28 + 14*14 + 7*7, 28*28)  # MNIST is 28x28
        self.reshape = nn.Sequential(
            nn.Linear(28*28, 28*28*1),  # Reshape to 28x28x1
            nn.Unflatten(1, (1, 28, 28))  # Convert to 4D tensor
        )

        # Second block: Multi-scale feature extraction paths
        # Path 1: 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Path 2: 1x1 + 3x3 + 3x3 convolutions
        self.path2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Path 3: 1x1 + 3x3 convolution
        self.path3 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Path 4: Average pooling + 1x1 convolution
        self.path4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1, 64, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Final layers
        self.fc2 = nn.Linear(256, 128)  # 4 paths * 64 channels
        self.fc3 = nn.Linear(128, 10)   # 10 classes for MNIST

    def forward(self, x):
        # First block processing
        pool1 = self.avg_pool1(x)
        pool2 = self.avg_pool2(x)
        pool3 = self.avg_pool3(x)

        # Flatten and concatenate
        flat1 = self.flatten(pool1)
        flat2 = self.flatten(pool2)
        flat3 = self.flatten(pool3)

        concat = torch.cat([flat1, flat2, flat3], dim=1)
        fc_out = self.fc1(concat)
        reshaped = self.reshape(fc_out)

        # Second block processing
        path1_out = self.path1(reshaped)
        path2_out = self.path2(reshaped)
        path3_out = self.path3(reshaped)
        path4_out = self.path4(reshaped)

        # Adaptive pooling to make all paths same size
        path1_out = F.adaptive_avg_pool2d(path1_out, (1, 1))
        path2_out = F.adaptive_avg_pool2d(path2_out, (1, 1))
        path3_out = F.adaptive_avg_pool2d(path3_out, (1, 1))
        path4_out = F.adaptive_avg_pool2d(path4_out, (1, 1))

        # Concatenate all paths
        concat_paths = torch.cat([path1_out, path2_out, path3_out, path4_out], dim=1)

        # Final classification
        flat_final = self.flatten(concat_paths)
        fc2_out = self.fc2(flat_final)
        output = self.fc3(fc2_out)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch of 1 MNIST image (1 channel, 28x28)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)