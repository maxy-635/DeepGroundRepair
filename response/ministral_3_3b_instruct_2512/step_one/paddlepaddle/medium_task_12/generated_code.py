import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: 3x3 conv with 64 channels
        self.block1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normalized_shape=64),
            nn.ReLU()
        )

        # Second block: 3x3 conv with 128 channels
        self.block2 = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normalized_shape=128),
            nn.ReLU()
        )

        # Third block: 3x3 conv with 256 channels
        self.block3 = nn.Sequential(
            nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normalized_shape=256),
            nn.ReLU()
        )

        # Flatten and fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=256 * 8 * 8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Process first block
        x = self.block1(x)

        # Process second block and concatenate with first block's output
        x2 = self.block2(x)
        x = paddle.concat([x, x2], axis=1)

        # Process third block and concatenate with second block's output
        x3 = self.block3(x)
        x = paddle.concat([x, x2], axis=1)

        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor with shape (batch_size, channels, height, width)
    # For CIFAR-10, height and width are 32
    input_tensor = paddle.randn((1, 3, 32, 32))

    # Forward pass
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)