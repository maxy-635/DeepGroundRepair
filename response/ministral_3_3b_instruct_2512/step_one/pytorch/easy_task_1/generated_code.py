import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Sequential layers: Conv2d -> MaxPool2d -> Conv2d
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
        # Flatten layer
        self.flatten = nn.Flatten()
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=64 * 7 * 7, out_features=128),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor with shape (batch_size=1, channels=1, height=28, width=28)
    input_tensor = torch.randn(1, 1, 28, 28)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)