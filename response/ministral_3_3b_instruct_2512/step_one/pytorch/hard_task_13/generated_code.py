import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: four parallel branches
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Concatenate the outputs of the four branches
        self.concat = nn.Sequential(
            nn.Conv2d(64 * 4, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Second block: Global average pooling and fully connected layers
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # First block: four parallel branches
        conv1x1_out = self.conv1x1(x)
        conv3x3_out = self.conv3x3(x)
        conv5x5_out = self.conv5x5(x)
        max_pool_out = self.max_pool(x)

        # Concatenate the outputs of the four branches
        concat_out = torch.cat((conv1x1_out, conv3x3_out, conv5x5_out, max_pool_out), dim=1)
        concat_out = self.concat(concat_out)

        # Second block: Global average pooling and fully connected layers
        pooled_features = self.avg_pool(concat_out)
        pooled_features = torch.flatten(pooled_features, 1)

        # Fully connected layers
        fc1_out = self.fc1(pooled_features)
        fc2_out = self.fc2(fc2_out)

        return fc2_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor with batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (1, 10)