import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the block
        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.DepthwiseConv2d(in_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2D(in_channels=out_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU()
            )

        # Create three branches
        self.branch1 = block(1, 16)
        self.branch2 = block(1, 32)
        self.branch3 = block(1, 64)

        # Concatenate the outputs
        self.concat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16*28*28 + 32*28*28 + 64*28*28, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        # Pass the input through the branches
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)

        # Concatenate the outputs
        out = paddle.concat([out1, out2, out3], axis=1)

        # Pass through the final layers
        out = self.concat(out)

        return out


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)