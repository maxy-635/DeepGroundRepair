import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Average Pooling and Flattening
        self.pooling_layers = nn.Sequential(
            nn.AvgPool2d(kernel_size=1, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=4, stride=4)
        )

        # Flatten and concatenate outputs
        self.flatten = nn.Flatten()
        self.concatenate = nn.Sequential(
            nn.Linear(28 * 28 * 3, 1024),  # Adjust input features based on pooling output
            nn.Reshape((4, 256))  # Reshape to 4D tensor
        )

        # Second block: Multi-path Feature Extraction
        self.path1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1, stride=1),
            nn.Dropout(0.2)
        )

        self.path2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1, stride=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.Dropout(0.2)
        )

        self.path3 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=1, stride=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.Dropout(0.2)
        )

        self.path4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1, 32, kernel_size=1, stride=1),
            nn.Dropout(0.2)
        )

        # Concatenate paths along channel dimension
        self.concat = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1),
            nn.Dropout(0.2)
        )

        # Fully connected layers for classification
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        # First block processing
        pooled_outputs = []
        for pool in self.pooling_layers:
            pooled_outputs.append(pool(x))
        pooled_outputs = torch.cat(pooled_outputs, dim=1)
        flattened = self.flatten(pooled_outputs)
        flattened_reshaped = self.concatenate(flattened)

        # Second block processing
        path1_out = self.path1(x)
        path2_out = self.path2(x)
        path3_out = self.path3(x)
        path4_out = self.path4(x)

        # Concatenate paths
        concatenated = torch.cat((path1_out, path2_out, path3_out, path4_out), dim=1)
        concatenated = self.concat(concatenated)

        # Fully connected layers
        x = torch.flatten(concatenated, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)