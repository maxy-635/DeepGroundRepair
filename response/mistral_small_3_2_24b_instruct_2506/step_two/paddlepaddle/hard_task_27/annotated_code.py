import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()
        # Depthwise separable convolution
        self.depthwise_conv = nn.Conv2D(
            in_channels=3,
            out_channels=3,
            kernel_size=7,
            stride=1,
            padding=3,
            groups=3  # groups=in_channels for depthwise convolution
        )
        self.pointwise_conv = nn.Conv2D(
            in_channels=3,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=[3, 32, 32])  # Assuming input size 32x32

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=3 * 32 * 32, out_features=3 * 32 * 32)
        self.fc2 = nn.Linear(in_features=3 * 32 * 32, out_features=3 * 32 * 32)

        # Final classification layers
        self.fc3 = nn.Linear(in_features=3 * 32 * 32, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Depthwise separable convolution
        x_depthwise = self.depthwise_conv(x)
        x_pointwise = self.pointwise_conv(x_depthwise)
        x_norm = self.layer_norm(x_pointwise)

        # Fully connected layers
        x_flat = paddle.flatten(x_norm, start_axis=1)
        x_fc1 = self.fc1(x_flat)
        x_fc2 = self.fc2(x_fc1)

        # Residual connection
        x_residual = paddle.add(x_flat, x_fc2)

        # Final classification
        x_fc3 = self.fc3(x_residual)
        x_fc4 = self.fc4(x_fc3)

        return x_fc4

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]