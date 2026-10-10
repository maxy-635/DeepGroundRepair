import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distribution import ReshapeTransform

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial feature extraction
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(num_features=32)
        self.relu1 = nn.ReLU()

        # Compression and weighted feature map generation
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.fc1 = nn.Linear(in_features=32, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        self.reshape = ReshapeTransform(in_event_shape=[32], out_event_shape=[32, 1, 1])

        # Feature concatenation
        # Dimensionality reduction and downsampling
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.avg_pool = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)

        # Classification
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(in_features=16384, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Initial feature extraction
        x = self.conv1(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.bn1(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.relu1(x)  # output tensor shape: (1, 32, 32, 32)

        # Compression and weighted feature map generation
        pooled_features = self.global_avg_pool(x)  # output tensor shape: (1, 32, 1, 1)
        pooled_features = self.flatten(pooled_features)  # output tensor shape: (1, 32)
        weighted_features = self.fc1(pooled_features)  # output tensor shape: (1, 32)
        weighted_features = self.relu1(weighted_features)  # output tensor shape: (1, 32)
        weighted_features = self.fc2(weighted_features)  # output tensor shape: (1, 32)
        weighted_features = self.reshape(weighted_features)  # output tensor shape: (1, 32, 1, 1)
        
        # Feature concatenation
        x = x * weighted_features  # output tensor shape: (1, 32, 32, 32)
        x = paddle.concat([x, x], axis=1)  # output tensor shape: (1, 64, 32, 32)

        # Dimensionality reduction and downsampling
        x = self.conv2(x)  # output tensor shape: (1, 64, 32, 32)
        x = self.avg_pool(x)  # output tensor shape: (1, 64, 16, 16)

        # Classification
        x = self.flatten(x)  # output tensor shape: (1, 16384)
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)