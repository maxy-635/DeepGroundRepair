import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Split input into three groups
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )

        # Concatenate the outputs
        self.concat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32 * 32 * 32 * 3, out_features=128),  # Adjust input features based on image size
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        # Split the input along the channel dimension
        group1 = self.conv1x1(x)
        group2 = self.conv3x3(x)
        group3 = self.conv5x5(x)

        # Concatenate the outputs
        fused_features = torch.concat([group1, group2, group3], dim=1)

        # Pass through fully connected layers
        output = self.concat(fused_features)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)