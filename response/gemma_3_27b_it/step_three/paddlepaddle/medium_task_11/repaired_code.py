import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        # Channel attention module
        self.avg_pool = nn.AvgPool2D(kernel_size=1, stride=1)
        self.max_pool = nn.MaxPool2D(kernel_size=1, stride=1)
        self.fc1_avg = nn.Linear(in_features=64, out_features=32)
        self.relu_avg = nn.ReLU()
        self.fc2_avg = nn.Linear(in_features=32, out_features=64)
        self.fc1_max = nn.Linear(in_features=64, out_features=32)
        self.relu_max = nn.ReLU()
        self.fc2_max = nn.Linear(in_features=32, out_features=64)
        self.sigmoid = nn.Sigmoid()

        # Spatial feature extraction
        self.avg_pool2d = nn.AvgPool2D(kernel_size=2, stride=2)
        self.max_pool2d = nn.MaxPool2D(kernel_size=2, stride=2)

        # Fully connected layer for final output
        self.flatten = nn.Flatten()
        self.fc_output = nn.Linear(in_features=64 * 8 * 8, out_features=10)  # Assuming input size 32x32

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Initial convolution
        x = self.conv1(x)  # output tensor shape: (1, 64, 32, 32)
        x = self.relu1(x)  # output tensor shape: (1, 64, 32, 32)

        # Channel attention
        avg_out = self.avg_pool(x)  # output tensor shape: (1, 64, 32, 32)
        avg_out = self.fc1_avg(avg_out)
        avg_out = self.relu_avg(avg_out)
        avg_out = self.fc2_avg(avg_out)

        max_out = self.max_pool(x)
        max_out = self.fc1_max(max_out)
        max_out = self.relu_max(max_out)
        max_out = self.fc2_max(max_out)

        channel_attention = self.sigmoid(avg_out + max_out)
        x = x * channel_attention

        # Spatial feature extraction
        avg_spatial = self.avg_pool2d(x)
        max_spatial = self.max_pool2d(x)

        # Fuse spatial features
        fused_features = paddle.concat([avg_spatial, max_spatial], axis=1)

        # Combine channel and spatial features
        x = x * fused_features

        # Flatten and output
        x = self.flatten(x)
        x = self.fc_output(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)