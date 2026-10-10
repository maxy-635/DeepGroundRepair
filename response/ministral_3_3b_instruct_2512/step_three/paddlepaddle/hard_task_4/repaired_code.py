import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Channel dimension tripling with 1x1 convolution
        self.conv1x1 = nn.Conv2D(in_channels=3, out_channels=9, kernel_size=1, stride=1, padding=0)

        # Initial feature extraction with 3x3 depthwise separable convolution
        self.depthwise_conv = nn.Conv2D(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1, groups=9)

        # Channel attention mechanism
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.fc1 = nn.Linear(in_features=9, out_features=9)
        self.fc2 = nn.Linear(in_features=9, out_features=9)

        # 1x1 convolution to reduce dimensionality
        self.conv_reduce = nn.Conv2D(in_channels=9, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Fully connected layer for classification
        self.flatten = nn.Flatten()
        self.fc_classification = nn.Linear(in_features=324, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Channel dimension tripling with 1x1 convolution
        x = self.conv1x1(x)  # output tensor shape: (1, 9, 32, 32)

        # Initial feature extraction with 3x3 depthwise separable convolution
        x = self.depthwise_conv(x)  # output tensor shape: (1, 9, 32, 32)

        # Channel attention mechanism
        channel_attention = self.avg_pool(x)  # output tensor shape: (1, 9, 1, 1)
        channel_attention = self.fc1(channel_attention)
        channel_attention = F.relu(channel_attention)
        channel_attention = self.fc2(channel_attention)
        channel_attention = F.log_softmax(channel_attention, axis=1)
        channel_attention = channel_attention.view(-1, 9, 1, 1)
        x = x * channel_attention

        # 1x1 convolution to reduce dimensionality
        x = self.conv_reduce(x)

        # Combine with initial input (assuming residual connection)
        x = x + x

        # Flatten and pass through fully connected layer for classification
        x = self.flatten(x)
        x = self.fc_classification(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Example input tensor for CIFAR-10 (batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)