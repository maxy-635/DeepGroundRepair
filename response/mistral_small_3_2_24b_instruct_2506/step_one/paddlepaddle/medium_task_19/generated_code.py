import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 convolution for dimensionality reduction
        self.branch1 = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2_conv1 = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.branch2_conv2 = nn.Conv2D(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Branch 3: 1x1 convolution followed by 5x5 convolution
        self.branch3_conv1 = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.branch3_conv2 = nn.Conv2D(
            in_channels=64,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2
        )

        # Branch 4: 3x3 max pooling followed by 1x1 convolution
        self.branch4_pool = nn.MaxPool2D(
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.branch4_conv = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Flatten and fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 4 * 32 * 32, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=10)

    def forward(self, x):
        # Branch 1
        out1 = self.branch1(x)

        # Branch 2
        out2 = self.branch2_conv1(x)
        out2 = self.branch2_conv2(out2)

        # Branch 3
        out3 = self.branch3_conv1(x)
        out3 = self.branch3_conv2(out3)

        # Branch 4
        out4 = self.branch4_pool(x)
        out4 = self.branch4_conv(out4)

        # Concatenate all branches
        out = paddle.concat([out1, out2, out3, out4], axis=1)

        # Flatten and fully connected layers
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # CIFAR-10 input shape: (batch_size, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]