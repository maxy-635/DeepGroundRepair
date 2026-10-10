import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: parallel branches
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(inplace=True)
        )

        # Concatenate the outputs
        self.concat = nn.Sequential()

        # Second block: feature processing
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Weight generation layers
        self.fc1 = nn.Linear(in_features='mask', out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=256)

        # Final classification layer
        self.fc_output = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First block: parallel processing
        conv1x1_out = self.conv1x1(x)  # output tensor shape: (1, 64, 32, 32)
        conv3x3_out = self.conv3x3(x)  # output tensor shape: (1, 64, 32, 32)
        conv5x5_out = self.conv5x5(x)  # output tensor shape: (1, 64, 32, 32)
        maxpool_out = self.maxpool(x)  # output tensor shape: (1, 3, 32, 32)

        # Concatenate along channel dimension
        concatenated = torch.cat([conv1x1_out, conv3x3_out, conv5x5_out, maxpool_out], dim=1)  # output tensor shape: (1, 195, 32, 32)

        # Second block: feature processing
        pooled = self.global_avg_pool(concatenated)  # output tensor shape: (1, 195, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # output tensor shape: (1, 195)

        # Generate weights
        weights = F.relu(self.fc1(pooled))
        weights = self.fc2(weights)

        # Reshape and multiply
        weights = weights.view(weights.size(0), 256, 1, 1)
        weighted_features = concatenated * weights

        # Final classification
        output = self.fc_output(weighted_features.mean(dim=[2, 3]))

        return output

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)