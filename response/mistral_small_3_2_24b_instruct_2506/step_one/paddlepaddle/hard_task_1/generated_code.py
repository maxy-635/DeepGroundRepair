import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Block 1: Channel attention
        self.avg_pool1 = nn.AdaptiveAvgPool2D(output_size=1)
        self.max_pool1 = nn.AdaptiveMaxPool2D(output_size=1)

        self.fc1_path1 = nn.Linear(in_features=64, out_features=64)
        self.fc2_path1 = nn.Linear(in_features=64, out_features=64)

        self.fc1_path2 = nn.Linear(in_features=64, out_features=64)
        self.fc2_path2 = nn.Linear(in_features=64, out_features=64)

        # Block 2: Spatial attention
        self.avg_pool2 = nn.AdaptiveAvgPool2D(output_size=1)
        self.max_pool2 = nn.AdaptiveMaxPool2D(output_size=1)

        self.conv2 = nn.Conv2D(
            in_channels=2,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Additional branch
        self.conv3 = nn.Conv2D(
            in_channels=64,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Final classification layer
        self.fc_final = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)

        # Block 1: Channel attention
        # Path 1
        path1 = self.avg_pool1(x)
        path1 = paddle.flatten(path1, start_axis=1)
        path1 = self.fc1_path1(path1)
        path1 = F.relu(path1)
        path1 = self.fc2_path1(path1)
        path1 = F.log_sigmoid(path1)

        # Path 2
        path2 = self.max_pool1(x)
        path2 = paddle.flatten(path2, start_axis=1)
        path2 = self.fc1_path2(path2)
        path2 = F.relu(path2)
        path2 = self.fc2_path2(path2)
        path2 = F.log_sigmoid(path2)

        # Combine paths
        channel_attention = path1 + path2
        channel_attention = F.relu(channel_attention)
        channel_attention = paddle.unsqueeze(channel_attention, axis=-1)
        channel_attention = paddle.unsqueeze(channel_attention, axis=-1)
        x = paddle.multiply(x, channel_attention)

        # Block 2: Spatial attention
        avg_out = self.avg_pool2(x)
        max_out = self.max_pool2(x)
        spatial_attention = paddle.concat([avg_out, max_out], axis=1)
        spatial_attention = self.conv2(spatial_attention)
        spatial_attention = F.sigmoid(spatial_attention)

        x = paddle.multiply(x, spatial_attention)

        # Additional branch
        x_branch = self.conv3(x)
        x = x + x_branch
        x = F.relu(x)

        # Global average pooling and classification
        x = self.avg_pool1(x)
        x = paddle.flatten(x, start_axis=1)
        x = self.fc_final(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected shape: [1, 10]