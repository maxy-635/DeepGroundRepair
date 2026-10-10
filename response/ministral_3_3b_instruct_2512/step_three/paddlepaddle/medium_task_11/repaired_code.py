import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolutional layer
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Channel attention path 1: Global Average Pooling + Fully Connected Layers
        self.avg_pool1 = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.fc1_avg = nn.Linear(in_features=64, out_features=64)
        self.fc2_avg = nn.Linear(in_features=64, out_features=1)

        # Channel attention path 2: Global Max Pooling + Fully Connected Layers
        self.max_pool1 = nn.AdaptiveMaxPool2D(output_size=(1, 1))
        self.fc1_max = nn.Linear(in_features=64, out_features=64)
        self.fc2_max = nn.Linear(in_features=64, out_features=1)

        # Spatial attention path: Average and Max Pooling + Concatenation
        self.avg_pool2 = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.max_pool2 = nn.AdaptiveMaxPool2D(output_size=(1, 1))

        # Fully connected layer for final output
        self.fc_final = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Initial convolutional layer
        x = self.conv1(x)

        # Channel attention path 1
        avg_pool_out = self.avg_pool1(x)
        avg_pool_out = paddle.reshape(avg_pool_out, shape=[-1, 64])
        avg_pool_out = self.fc1_avg(avg_pool_out)
        avg_pool_out = self.fc2_avg(avg_pool_out)
        avg_pool_out = paddle.reshape(avg_pool_out, shape=[-1, 1, 1, 1])

        # Channel attention path 2
        max_pool_out = self.max_pool1(x)
        max_pool_out = paddle.reshape(max_pool_out, shape=[-1, 64])
        max_pool_out = self.fc1_max(max_pool_out)
        max_pool_out = self.fc2_max(max_pool_out)
        max_pool_out = paddle.reshape(max_pool_out, shape=[-1, 1, 1, 1])

        # Channel attention weights
        channel_attention_weights = avg_pool_out + max_pool_out
        channel_attention_weights = F.gelu(channel_attention_weights)
        x = x * channel_attention_weights

        # Spatial attention path
        avg_spatial_out = self.avg_pool2(x)
        max_spatial_out = self.max_pool2(x)

        # Concatenate spatial features along channel dimension
        spatial_features = paddle.concat([avg_spatial_out, max_spatial_out], axis=1)

        # Combine channel and spatial features
        x = x * spatial_features

        # Flatten and pass through final fully connected layer
        x = paddle.reshape(x, shape=[-1, 128])
        x = self.fc_final(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # Example input tensor for CIFAR-10
    output_tensor = model(input_tensor)
    print("Input tensor shape:", input_tensor.shape)
    print("Output tensor shape:", output_tensor.shape)