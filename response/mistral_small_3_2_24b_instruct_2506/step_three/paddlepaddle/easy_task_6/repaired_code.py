import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        # Branch path (identity connection)
        self.branch = nn.Sequential()

    def forward(self, x):
        # Main path
        out = self.conv1(x)
        out = self.conv2(out)

        # Branch path
        branch_out = self.branch(x)

        # Combine paths
        out = paddle.add(out, branch_out)

        # Flatten and fully connected layer
        out = paddle.flatten(out, start_axis=1)
        out = nn.Linear(in_features=28*28, out_features=10)(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]