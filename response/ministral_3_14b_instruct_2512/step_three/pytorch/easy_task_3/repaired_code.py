import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: two conv layers followed by max pooling
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Second block: four conv layers followed by max pooling
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64*7*7, out_features=512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=512),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=256),
            nn.Linear(in_features=256, out_features=10)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.fc_layers(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(32, 1, 28, 28)  # Batch size of 32, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])