import paddle
import paddle.nn as nn
import paddle.nn.functional as F

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

        # Channel attention mechanism
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.max_pool = nn.AdaptiveMaxPool2D(output_size=1)

        self.fc1_avg = nn.Linear(in_features=64, out_features=32)
        self.fc2_avg = nn.Linear(in_features=32, out_features=64)

        self.fc1_max = nn.Linear(in_features=64, out_features=32)
        self.fc2_max = nn.Linear(in_features=32, out_features=64)

        # Spatial attention mechanism
        self.avg_pool_spatial = nn.AdaptiveAvgPool2D(output_size=1)
        self.max_pool_spatial = nn.AdaptiveMaxPool2D(output_size=1)

        # Final fully connected layer
        self.fc_final = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)

        # Channel attention
        avg_out = self.avg_pool(x)
        avg_out = avg_out.flatten(1)
        avg_out = self.fc1_avg(avg_out)
        avg_out = F.relu(avg_out)
        avg_out = self.fc2_avg(avg_out)
        avg_out = avg_out.unsqueeze(-1).unsqueeze(-1)

        max_out = self.max_pool(x)
        max_out = max_out.flatten(1)
        max_out = self.fc1_max(max_out)
        max_out = F.relu(max_out)
        max_out = self.fc2_max(max_out)
        max_out = max_out.unsqueeze(-1).unsqueeze(-1)

        channel_attention = paddle.add(avg_out, max_out)
        channel_attention = F.relu(channel_attention)
        x = paddle.multiply(x, channel_attention)

        # Spatial attention
        avg_spatial = self.avg_pool_spatial(x)
        max_spatial = self.max_pool_spatial(x)
        spatial_attention = paddle.concat([avg_spatial, max_spatial], axis=1)

        # Combine channel and spatial attention
        x = paddle.multiply(x, spatial_attention)

        # Final classification
        x = x.flatten(1)
        x = self.fc_final(x)
        x = F.softmax(x, axis=1)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((32, 3, 32, 32))  # batch_size=32, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [32, 10]