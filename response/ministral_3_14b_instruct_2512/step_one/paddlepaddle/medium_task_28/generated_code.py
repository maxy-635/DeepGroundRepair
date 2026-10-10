import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Attention mechanism components
        self.attention_conv = nn.Conv2D(
            in_channels=3,  # Assuming CIFAR-10 input has 3 channels
            out_channels=3,  # Same number of channels for attention weights
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.attention_softmax = nn.Softmax(axis=1)

        # Dimensionality reduction components
        self.reduce_conv = nn.Conv2D(
            in_channels=3,
            out_channels=1,  # One-third of original 3 channels
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=[3, 32, 32])  # Assuming 32x32 input size
        self.relu = nn.ReLU()

        # Dimensionality restoration components
        self.restore_conv = nn.Conv2D(
            in_channels=1,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Classification components
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=3*32*32, out_features=10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        # Generate attention weights
        attention_weights = self.attention_conv(x)
        attention_weights = self.attention_softmax(attention_weights)

        # Apply attention to input
        attended_features = paddle.multiply(x, attention_weights)

        # Reduce dimensionality
        reduced = self.reduce_conv(attended_features)
        normalized = self.layer_norm(reduced)
        activated = self.relu(normalized)

        # Restore dimensionality
        restored = self.restore_conv(activated)

        # Add to original input
        residual = x + restored

        # Classification
        flattened = self.flatten(residual)
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, channels, height, width]
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)