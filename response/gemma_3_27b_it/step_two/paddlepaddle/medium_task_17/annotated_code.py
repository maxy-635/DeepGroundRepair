import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(num_features=32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(num_features=64)
        self.relu2 = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=64, out_features=10)
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Reshape for channel shuffling
        original_shape = x.shape  # output tensor shape: (unknown)
        x = x.reshape('mask', 'mask', 'mask', 'mask')
        # Channel Shuffle (permutation)
        x = x.transpose((0, 1, 3, 2))
        x = x.reshape(original_shape)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))  # Dummy input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)