import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Depthwise separable convolution with 7x7 kernel
        self.depthwise_conv = nn.Conv2D(
            in_channels=3,
            out_channels=3,
            kernel_size=7,
            stride=1,
            padding=3,
            groups=3  # Depthwise convolution
        )
        self.pointwise_conv = nn.Conv2D(
            in_channels=3,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.layer_norm = nn.LayerNorm([3, 32, 32])  # Assuming input size is 32x32

        # Fully connected layers with same number of channels as input
        self.fc1 = nn.Linear(3 * 32 * 32, 3 * 32 * 32)
        self.fc2 = nn.Linear(3 * 32 * 32, 3 * 32 * 32)

        # Final classification layers
        self.fc3 = nn.Linear(3 * 32 * 32, 1024)
        self.fc4 = nn.Linear(1024, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Extract spatial features with depthwise separable convolution
        x_depthwise = self.depthwise_conv(x)
        x_pointwise = self.pointwise_conv(x_depthwise)
        x_norm = self.layer_norm(x_pointwise)

        # Flatten for fully connected layers
        x_flat = paddle.flatten(x_norm, start_axis=1)

        # Channel-wise feature transformation
        x_fc1 = self.fc1(x_flat)
        x_fc2 = self.fc2(x_fc1)

        # Combine original input with processed features
        x_original_flat = paddle.flatten(x, start_axis=1)
        x_combined = x_original_flat + x_fc2

        # Final classification
        x_class1 = self.fc3(x_combined)
        x_class2 = self.fc4(self.relu(x_class1))

        return x_class2

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)