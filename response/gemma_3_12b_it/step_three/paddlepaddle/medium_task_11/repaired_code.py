import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        # Channel attention path 1: Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.fc1_avg = nn.Linear(in_features=64, out_features=32)
        self.fc2_avg = nn.Linear(in_features=32, out_features=16)

        # Channel attention path 2: Global Max Pooling
        self.max_pool = nn.AdaptiveMaxPool2D(output_size=1)
        self.fc1_max = nn.Linear(in_features=64, out_features=32)
        self.fc2_max = nn.Linear(in_features=32, out_features=16)

        # Combine channel attention paths
        self.add_channel = lambda x, y: x + y  # Define a lambda function for addition

        # Spatial attention path: Average Pooling
        self.avg_pool_spatial = nn.AdaptiveAvgPool2D(output_size=1)

        # Spatial attention path: Max Pooling
        self.max_pool_spatial = nn.AdaptiveMaxPool2D(output_size=1)

        # Concatenate spatial features
        self.concat_spatial = lambda x, y: paddle.concat([x, y], axis=1)

        # Flatten and final fully connected layer
        self.flatten = nn.Flatten()
        self.fc_final = nn.Linear(in_features=640, out_features=10)  # Assuming 10 classes in CIFAR-10

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Initial convolutional layer
        x = self.conv1(x)  # output tensor shape: (1, 64, 32, 32)
        x = self.relu1(x)  # output tensor shape: (1, 64, 32, 32)

        # Channel attention path 1
        avg_out = self.avg_pool(x)  # output tensor shape: (1, 64, 1, 1)
        avg_out = self.fc1_avg(avg_out)
        avg_out = self.relu1(avg_out)
        avg_out = self.fc2_avg(avg_out)

        # Channel attention path 2
        max_out = self.max_pool(x)
        max_out = self.fc1_max(max_out)
        max_out = self.relu1(max_out)
        max_out = self.fc2_max(max_out)

        # Combine channel attention paths
        channel_attention = self.add_channel(avg_out, max_out)

        # Apply channel attention
        x = paddle.multiply(x, channel_attention)

        # Spatial attention path
        spatial_avg = self.avg_pool_spatial(x)
        spatial_max = self.max_pool_spatial(x)

        # Concatenate spatial features
        spatial_features = self.concat_spatial(spatial_avg, spatial_max)

        # Combine channel and spatial features
        combined_features = paddle.multiply(x, spatial_features)

        # Flatten and final fully connected layer
        x = self.flatten(combined_features)
        x = self.fc_final(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.rand(shape=[1, 3, 32, 32])  # Dummy input tensor for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)