import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Parallel branches
        self.conv1x1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.maxpool = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)

        # Block 2: Dimensionality reduction and weighted feature enhancement
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=64)  # Assuming input size 32x32
        self.fc2 = nn.Linear(in_features=64, out_features=64 * 64)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(in_features=64*64, out_features=10)
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1: Parallel branches
        branch1 = self.conv1x1(x)  # output tensor shape: (1, 64, 32, 32)
        branch2 = self.conv3x3(x)  # output tensor shape: (1, 64, 32, 32)
        branch3 = self.conv5x5(x)  # output tensor shape: (1, 64, 32, 32)
        branch4 = self.maxpool(x)  # output tensor shape of this layer should be (1, 3, 32, 32)

        # Concatenate the outputs of the branches
        x = paddle.concat([branch1, branch2, branch3, branch4], axis=1)

        # Block 2: Dimensionality reduction and weighted feature enhancement
        x = self.global_avg_pool(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        weights = self.fc1(x)
        weights = self.relu(weights)
        weights = self.fc2(weights)
        weights = paddle.reshape(weights, [x.shape[0], 64, 1, 1])
        x = x * weights
        x = self.fc3(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]