import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.CELU(alpha=1.0)
        )

        # Path 2: 1x1 -> 3x3 -> 3x3 convolutions
        self.path2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.CELU(alpha=1.0),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.CELU(alpha=1.0),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.CELU(alpha=1.0)
        )

        # Path 3: 1x1 -> 3x3 convolution
        self.path3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.CELU(alpha=1.0),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.CELU(alpha=1.0)
        )

        # Path 4: Max pooling -> 1x1 convolution
        self.path4 = nn.Sequential(
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0),
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.CELU(alpha=1.0)
        )

        # Concatenate paths
        self.concat = nn.Sequential(
            nn.Flatten(),
            nn.Dense(in_channels=32 * 4, out_channels=128),
            nn.CELU(alpha=1.0),
            nn.Softmax(axis=-1)
        )

    def forward(self, x):
        # Pass through parallel paths
        path1_out = self.path1(x)
        path2_out = self.path2(x)
        path3_out = self.path3(x)
        path4_out = self.path4(x)

        # Concatenate outputs
        concatenated = paddle.concat([path1_out, path2_out, path3_out, path4_out], axis=1)

        # Pass through dense and softmax layers
        output = self.concat(concatenated)
        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (CIFAR-10 image size is 32x32x3)
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)