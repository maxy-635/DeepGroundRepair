import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Split the input image into three channel groups
        self.conv1_1x1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv1_3x3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv1_5x5 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Concatenate the outputs of the three convolutional groups
        self.concat = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 512),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Extract features from each channel group
        x1 = self.conv1_1x1(x)
        x2 = self.conv1_3x3(x)
        x3 = self.conv1_5x5(x)

        # Concatenate the outputs
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.concat(x)

        # Pass through fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)