import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial 1x1 convolution to increase channel dimensionality
        self.conv1x1_expand = nn.Conv2D(in_channels=3, out_channels=9, kernel_size=1, stride=1, padding=0)

        # Depthwise separable convolution for feature extraction
        self.depthwise_conv = nn.Conv2D(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1, groups=9)

        # Channel attention mechanism
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.fc1 = nn.Linear(in_features=9, out_features=9//2)
        self.fc2 = nn.Linear(in_features=9//2, out_features=9)
        self.sigmoid = nn.Sigmoid()

        # 1x1 convolution to reduce dimensionality
        self.conv1x1_reduce = nn.Conv2D(in_channels=9, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Final classification layers
        self.flatten = nn.Flatten()
        self.fc_final = nn.Linear(in_features=3*32*32, out_features=10)  # Assuming input size is 32x32

    def forward(self, x):
        # Initial expansion
        x_expanded = self.conv1x1_expand(x)

        # Feature extraction
        features = self.depthwise_conv(x_expanded)

        # Channel attention
        pooled = self.global_avg_pool(features)
        pooled = self.flatten(pooled)
        attention_weights = self.fc2(self.fc1(pooled))
        attention_weights = self.sigmoid(attention_weights)
        attention_weights = paddle.reshape(attention_weights, shape=[-1, 9, 1, 1])
        weighted_features = paddle.multiply(features, attention_weights)

        # Dimensionality reduction
        reduced = self.conv1x1_reduce(weighted_features)

        # Combine with original input
        combined = paddle.add(reduced, x)

        # Final classification
        flattened = self.flatten(combined)
        output = self.fc_final(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]