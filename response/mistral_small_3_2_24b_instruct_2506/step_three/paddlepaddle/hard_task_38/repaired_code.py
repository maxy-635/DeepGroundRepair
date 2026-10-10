import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class ResidualBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)






        self.bn = nn.BatchNorm2D(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # the initial shape of x is: (32, 1, 28, 28)
        residual = x  # output tensor shape: (32, 33, 28, 28)
        out = self.conv(x)  # output tensor shape: (32, 32, 28, 28)
        out = self.bn(out)  # output tensor shape: (32, 32, 28, 28)
        out = self.relu(out)  # output tensor shape: (32, 32, 28, 28)
        out = paddle.concat([residual, out], axis=1)  # output tensor shape: (32, 33, 28, 28)
        return out

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First pathway
        self.pathway1 = nn.Sequential(
            ResidualBlock(in_channels=1, out_channels=32),
            ResidualBlock(in_channels=64, out_channels=32),
            ResidualBlock(in_channels=96, out_channels=32)
        )
        # Second pathway
        self.pathway2 = nn.Sequential(
            ResidualBlock(in_channels=1, out_channels=64),
            ResidualBlock(in_channels=128, out_channels=64),
            ResidualBlock(in_channels=192, out_channels=64)
        )
        # Classification head
        self.fc1 = nn.Linear(in_features=256 * 7 * 7, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (32, 1, 28, 28)
        # Process through both pathways
        out1 = self.pathway1(x)
        out2 = self.pathway2(x)
        # Concatenate outputs
        out = paddle.concat([out1, out2], axis=1)
        # Flatten and classify
        out = paddle.flatten(out, start_axis=1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[32, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [32, 10]