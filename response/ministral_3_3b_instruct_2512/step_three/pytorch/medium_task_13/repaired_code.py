import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the convolutional layers with concatenation
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Flatten the output of the convolutional layers
        self.flatten = nn.Flatten()

        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Concatenate the input with the output of each convolutional layer
        x = self.conv_layers[0](x)  # output tensor shape: (1, 32, 32, 32)
        x = torch.cat([x, self.conv_layers[0](x).squeeze(3)], dim=1)
        x = self.conv_layers[1](x)
        x = torch.cat([x, self.conv_layers[1](x)], dim=1)
        x = self.conv_layers[2](x)
        x = torch.cat([x, self.conv_layers[2](x)], dim=1)

        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = self.fc_layers[0](x)
        x = self.fc_layers[1](x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor with batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])