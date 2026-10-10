import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.depthwise_conv = nn.Conv2D(
            in_channels=3,
            out_channels=3,
            kernel_size=7,
            stride=1,
            padding=3
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=[3, 32, 32])  # Assuming input size is 32x32 for CIFAR-10
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
        self.branch = nn.Sequential()

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=3 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # Main path
        main = self.depthwise_conv(x)
        main = self.layer_norm(main)
        main = self.pointwise_conv1(main)
        main = self.pointwise_conv2(main)

        # Branch path
        branch = self.branch(x)

        # Combine paths
        combined = paddle.add(main, branch)

        # Classification
        combined = self.flatten(combined)
        combined = self.fc1(combined)
        combined = self.fc2(combined)

        return combined

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected shape: [1, 10]