import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(num_features=64)
        self.relu1 = nn.ReLU()

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)

        # 1x1 convolution for dimensionality reduction
        self.conv1x1 = nn.Conv2D(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')

        # Average pooling for downsampling
        self.avg_pool2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)

        # Final fully connected layer for classification
        self.fc3 = nn.Linear(in_features=32 * 8 * 8, out_features=10)  # Assuming input size is 32x32

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Initial feature extraction
        x = self.conv1(x)  # output tensor shape: (1, 64, 32, 32)
        x = self.bn1(x)  # output tensor shape: (1, 64, 32, 32)
        x = self.relu1(x)  # output tensor shape: (1, 64, 32, 32)

        # Global average pooling
        x_pool = self.avg_pool(x)  # output tensor shape: (1, 64, 1, 1)
        x_pool = paddle.reshape(x_pool, [x_pool.shape[0], -1])  # output tensor shape: (1, 64)

        # Fully connected layers
        x_pool = self.fc1(x_pool)  # output tensor shape: (1, 64)
        x_pool = self.fc2(x_pool)  # output tensor shape: (1, 64)
        x_pool = paddle.reshape(x_pool, [x_pool.shape[0], 64, 1, 1])  # output tensor shape: (1, 64, 1, 1)

        # Weighted feature maps
        x_weighted = x * x_pool  # output tensor shape: (1, 64, 32, 32)

        # Concatenate with input
        x_concat = paddle.concat([x, x_weighted], axis=1)  # output tensor shape: (1, 128, 32, 32)

        # Dimensionality reduction and downsampling
        x_reduced = self.conv1x1(x_concat)
        x_reduced = self.avg_pool2(x_reduced)

        # Flatten and final classification
        x_flat = paddle.reshape(x_reduced, [x_reduced.shape[0], -1])
        x_out = self.fc3(x_flat)

        return x_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)