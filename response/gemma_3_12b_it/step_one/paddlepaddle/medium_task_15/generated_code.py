import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolutional layer
        self.conv1 = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn1 = nn.BatchNorm2D(
            num_features=64
        )
        self.relu1 = nn.ReLU()

        # Global average pooling
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2D(
            output_size=1
        )

        # Fully connected layers
        self.fc1 = nn.Linear(
            in_features=64,
            out_features=64
        )
        self.fc2 = nn.Linear(
            in_features=64,
            out_features=3
        )

        # 1x1 convolution and average pooling
        self.conv1x1 = nn.Conv2D(
            in_channels=3,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.avg_pool = nn.AvgPool2D(
            kernel_size=2,
            stride=2,
            padding=0
        )

        # Final fully connected layer
        self.fc_out = nn.Linear(
            in_features=3,
            out_features=10
        )
        self.softmax = nn.Softmax(axis=-1)


    def forward(self, x):
        # Convolutional layer, batch normalization, and ReLU activation
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Global average pooling
        x = self.adaptive_avg_pool(x)

        # Flatten the feature maps
        x = paddle.flatten(x, [1, 2])

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)

        # Reshape to match initial feature size
        x = paddle.reshape(x, [x.shape[0], 3, 1, 1])

        # Multiply with initial features
        x = x * self.relu1(self.conv1(x))  # Use the output of conv1 again

        # Concatenate with input layer
        x = paddle.concat([x, self.relu1(self.conv1(x))], axis=1)
        
        # 1x1 convolution and average pooling
        x = self.conv1x1(x)
        x = self.avg_pool(x)

        # Flatten for fully connected layer
        x = paddle.flatten(x, [1, 2, 3])

        # Final fully connected layer
        x = self.fc_out(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[32, 3, 32, 32])  # Batch size of 32, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)