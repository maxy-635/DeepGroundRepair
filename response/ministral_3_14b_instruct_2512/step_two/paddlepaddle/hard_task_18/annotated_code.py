import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: feature extraction with two 3x3 conv layers and avg pooling
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=64, out_channels='mask', kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size='mask')

        # Second block: channel attention mechanism
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)

        # Final classification layer
        self.fc_classifier = nn.Linear(in_features=64*32*32, out_features=10)  # Assuming CIFAR-10 output

        # Activation functions
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(normalized_shape=[64])

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First block: feature extraction
        x_conv1 = self.relu(self.conv1(x))  # output tensor shape: (1, 64, 32, 32)
        x_conv2 = self.relu(self.conv2(x_conv1))  # output tensor shape of this layer should be (1, 3, 32, 32)
        x_pool = self.avg_pool(x_conv2)  # output tensor shape of this layer should be (1, 64, 32, 32)

        # Main path: add input to first block output
        x_main = x + x_pool

        # Second block: channel attention
        # Global average pooling to get channel weights
        channel_weights = self.global_avg_pool(x_main)
        channel_weights = channel_weights.squeeze(-1).squeeze(-1)
        channel_weights = self.relu(self.fc1(channel_weights))
        channel_weights = self.fc2(channel_weights)
        channel_weights = self.layer_norm(channel_weights)
        channel_weights = channel_weights.unsqueeze(-1).unsqueeze(-1)

        # Apply channel weights
        x_weighted = x_main * channel_weights

        # Flatten and classify
        x_flat = x_weighted.flatten(start_dim=1)
        output = self.fc_classifier(x_flat)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch of 1, 3 channels, 32x32 CIFAR-10 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)