import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: 1x1 convolution
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu1 = nn.ReLU()

        # Path 2: Two 3x3 convolutions stacked
        self.conv2_1 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2D(32)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2D(32)
        self.relu2_2 = nn.ReLU()

        # Path 3: Single 3x3 convolution
        self.conv3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2D(32)
        self.relu3 = nn.ReLU()

        # Path 4: Max pooling and 1x1 convolution
        self.pool4 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.conv4_1 = nn.Conv2D(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.bn4_1 = nn.BatchNorm2D(16)
        self.relu4_1 = nn.ReLU()

        # Concatenate the paths
        self.concat_dim1 = nn.Concat(dim=1)

        # Flatten
        self.flatten = nn.Flatten()

        # Dense layer
        self.dense = nn.Linear(16 * 64 * 32, 128)
        self.bn_dense = nn.BatchNorm2D(128)
        self.relu_dense = nn.ReLU()

        # Output layer
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        # Path 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Path 2
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)

        # Path 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Path 4
        x = self.pool4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu4_1(x)

        # Concatenate
        x = self.concat_dim1([x, x, x, x])

        # Flatten
        x = self.flatten(x)

        # Dense layer
        x = self.dense(x)
        x = self.bn_dense(x)
        x = self.relu_dense(x)

        # Output layer
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = paddle.randn(batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)