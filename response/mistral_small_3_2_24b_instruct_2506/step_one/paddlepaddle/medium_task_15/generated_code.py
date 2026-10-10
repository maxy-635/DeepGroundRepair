import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolutional layer
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(num_features=64)
        self.relu1 = nn.ReLU()

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=64)

        # 1x1 convolution and average pooling
        self.conv2 = nn.Conv2D(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.avg_pool2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)

        # Final fully connected layer for classification
        self.fc3 = nn.Linear(in_features=64 * 4 * 4, out_features=10)  # Assuming input size is 32x32 after pooling

    def forward(self, x):
        # Initial feature extraction
        initial_features = self.conv1(x)
        initial_features = self.bn1(initial_features)
        initial_features = self.relu1(initial_features)

        # Compress features
        compressed_features = self.avg_pool(initial_features)
        compressed_features = paddle.reshape(compressed_features, shape=[-1, 64])
        compressed_features = self.fc1(compressed_features)
        compressed_features = self.fc2(compressed_features)
        compressed_features = paddle.reshape(compressed_features, shape=[-1, 64, 1, 1])

        # Weighted feature maps
        weighted_features = paddle.multiply(x=initial_features, y=compressed_features)
        concatenated_features = paddle.concat(x=[initial_features, weighted_features], axis=1)

        # Reduce dimensionality and downsample
        reduced_features = self.conv2(concatenated_features)
        reduced_features = self.avg_pool2(reduced_features)
        reduced_features = paddle.reshape(reduced_features, shape=[-1, 64 * 4 * 4])

        # Final classification
        output = self.fc3(reduced_features)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]