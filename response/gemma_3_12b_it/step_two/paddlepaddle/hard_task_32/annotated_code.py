import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the specialized block
        self.block = nn.Sequential(
            nn.DepthwiseConv2D(in_channels=1, kernel_size=3, stride=1, padding=1, weight_attr=nn.initializer.KahemingNormal()),
            nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0, weight_attr=nn.initializer.KahemingNormal()),
            nn.Dropout(p=0.2),
            nn.DepthwiseConv2D(in_channels=64, kernel_size=3, stride=1, padding=1, weight_attr=nn.initializer.KahemingNormal()),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, weight_attr=nn.initializer.KahemingNormal()),
            nn.Dropout(p=0.2)
        )

        # Create three branches
        self.branch1 = self.block
        self.branch2 = self.block
        self.branch3 = self.block

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 3, out_features=128, weight_attr=nn.initializer.KahemingNormal())
        self.fc2 = nn.Linear(in_features=128, out_features=10, weight_attr=nn.initializer.KahemingNormal())

    def forward(self, x):
        # Pass the input through the three branches
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)

        # Concatenate the outputs
        out = paddle.concat([out1, out2, out3], axis=1)

        # Pass through the fully connected layers
        out = F.leaky_relu(self.fc1(out), negative_slope=0.2)
        out = self.fc2(out)

        return out


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for MNIST (1, 1, 28, 28)
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)