import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Channel attention branch
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.max_pool = nn.AdaptiveMaxPool2D(output_size=(1, 1))
        self.fc1_avg = nn.Linear(in_features=64, out_features=32)
        self.fc2_avg = nn.Linear(in_features=32, out_features=64)
        self.fc1_max = nn.Linear(in_features=64, out_features=32)
        self.fc2_max = nn.Linear(in_features=32, out_features=64)
        self.sigmoid = nn.Sigmoid()

        # Spatial attention branch
        self.avg_pool_spatial = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.max_pool_spatial = nn.AdaptiveMaxPool2D(output_size=(1, 1))
        self.conv1d_avg = nn.Conv2D(in_channels=64, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.conv1d_max = nn.Conv2D(in_channels=64, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid_spatial = nn.Sigmoid()

        # Final layers
        self.flatten = nn.Flatten()
        self.fc_final = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = F.relu(x)

        # Channel attention
        avg_out = self.avg_pool(x)
        avg_out = avg_out.reshape([avg_out.shape[0], -1])
        avg_out = self.fc1_avg(avg_out)
        avg_out = F.relu(avg_out)
        avg_out = self.fc2_avg(avg_out)

        max_out = self.max_pool(x)
        max_out = max_out.reshape([max_out.shape[0], -1])
        max_out = self.fc1_max(max_out)
        max_out = F.relu(max_out)
        max_out = self.fc2_max(max_out)

        channel_attention = avg_out + max_out
        channel_attention = self.sigmoid(channel_attention)
        channel_attention = channel_attention.reshape([channel_attention.shape[0], 1, 1, 1])
        x = paddle.multiply(x, channel_attention)

        # Spatial attention
        avg_spatial = self.avg_pool_spatial(x)
        max_spatial = self.max_pool_spatial(x)

        avg_spatial = self.conv1d_avg(avg_spatial)
        max_spatial = self.conv1d_max(max_spatial)

        spatial_attention = avg_spatial + max_spatial
        spatial_attention = self.sigmoid_spatial(spatial_attention)
        spatial_attention = spatial_attention.reshape([spatial_attention.shape[0], 1, spatial_attention.shape[2], spatial_attention.shape[3]])

        x = paddle.multiply(x, spatial_attention)

        # Final layers
        x = self.flatten(x)
        x = self.fc_final(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([32, 3, 32, 32])  # Batch of 32 CIFAR-10 images (3 channels, 32x32)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Should be [32, 10]