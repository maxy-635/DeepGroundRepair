import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.depthwise_conv = nn.Conv2D(
            in_channels=3,
            out_channels=3,
            kernel_size=7,
            stride=1,
            padding=3,
            groups=3  # Depthwise convolution
        )

        self.layer_norm = nn.LayerNorm([3, 32, 32])  # Assuming CIFAR-10 input size 32x32

        # Two 1x1 pointwise convolutions
        self.pointwise_conv1 = nn.Conv2D(
            in_channels=3,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.pointwise_conv2 = nn.Conv2D(
            in_channels=3,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Branch path (identity connection)
        self.branch = nn.Identity()

        # Flatten and fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=3*32*32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        # Main path
        main_path = self.depthwise_conv(x)
        main_path = self.layer_norm(main_path)
        main_path = self.pointwise_conv1(main_path)
        main_path = self.pointwise_conv2(main_path)

        # Branch path
        branch_path = self.branch(x)

        # Combine paths
        combined = main_path + branch_path

        # Classification head
        combined = self.flatten(combined)
        combined = self.fc1(combined)
        combined = F.relu(combined)
        combined = self.fc2(combined)

        return combined

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, channels, height, width]
    input_tensor = paddle.randn((1, 3, 32, 32))  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Should be [1, 10]