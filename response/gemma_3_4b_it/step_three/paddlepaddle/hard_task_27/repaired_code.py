import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Spatial feature extraction
        self.depthwise_conv = nn.Conv2D(
            in_channels=3,  # CIFAR-10 input channels
            out_channels=16,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        self.layer_norm1 = nn.LayerNorm(normalized_shape=1)

        # Fully connected layers
        self.fc1 = paddle.nn.fc.Linear(
            in_features=16 * 7 * 7,
            out_features=128,
            bias_init=False,
            name="fc1"
        )
        self.layer_norm2 = nn.LayerNorm(normalized_shape=1)
        self.fc2 = paddle.nn.fc.Linear(
            in_features=128,
            out_features=10,
            bias_init=False,
            name="fc2"
        )

    def forward(self, x):
        # Spatial feature extraction
        x = self.depthwise_conv(x)
        x = self.layer_norm1(x)

        # Combine original input with processed features
        x = paddle.add(x, self.fc1(x))
        x = self.layer_norm2(x)

        # Output classification
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for validation
    input_tensor = paddle.randn(batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)