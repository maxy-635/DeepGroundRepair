import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Get input shape (assuming CIFAR-10: 3x32x32)
        self.input_shape = (3, 32, 32)

        # Calculate channels_per_group
        channels = self.input_shape[0]
        groups = 3
        channels_per_group = channels // groups

        # Channel Shuffle layer
        self.channel_shuffle = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 32 * 32, channels * 32 * 32),
            nn.Unflatten(1, (channels, 32, 32)),
            nn.Unflatten(2, (32, 32, channels))
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 10),  # 10 classes for CIFAR-10
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Get input shape
        batch_size, channels, height, width = x.shape

        # Reshape to (height, width, groups, channels_per_group)
        x = x.permute(0, 2, 3, 1)  # (batch, height, width, channels)
        x = x.reshape(batch_size, height, width, 3, channels//3)

        # Permute to (height, width, channels_per_group, groups)
        x = x.permute(0, 1, 2, 4, 3)

        # Reshape back to original input shape
        x = x.reshape(batch_size, height, width, channels)
        x = x.permute(0, 3, 1, 2)  # (batch, channels, height, width)

        # Apply channel shuffle
        x = self.channel_shuffle(x)

        # Classification
        output = self.classifier(x)

        return output

if __name__ == "__main__":
    model = DL_Model()

    # Create dummy input tensor (batch_size=4, channels=3, height=32, width=32)
    input_tensor = torch.randn(4, 3, 32, 32)

    # Forward pass
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)