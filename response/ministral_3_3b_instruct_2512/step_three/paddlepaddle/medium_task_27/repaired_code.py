import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: 3x3 convolutional branch
        self.conv3x3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer_norm3x3 = nn.LayerNorm(normalized_shape=(64,))

        # Branch 2: 5x5 convolutional branch
        self.conv5x5 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.layer_norm5x5 = nn.LayerNorm(normalized_shape=64)

        # Global average pooling
        self.avg_pool = nn.AvgPool2D(kernel_size=8, stride=1, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

        # Attention weights
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Branch 1: 3x3 convolutional branch
        conv3x3_out = self.conv3x3(x)  # output tensor shape: (1, 64, 32, 32)
        conv3x3_out = self.layer_norm3x3(conv3x3_out)
        conv3x3_out = paddle.relu(conv3x3_out)

        # Branch 2: 5x5 convolutional convolutional branch
        conv5x5_out = self.conv5x5(x)
        conv5x5_out = self.layer_norm5x5(conv5x5_out)
        conv5x5_out = paddle.relu(conv5x5_out)

        # Combine features through addition
        combined_features = paddle.add(conv3x3_out, conv5x5_out)

        # Global average pooling
        pooled_features = self.avg_pool(combined_features)

        # Flatten the pooled features
        pooled_features = paddle.flatten(pooled_features)

        # Apply attention weights
        attention_weights = self.softmax(pooled_features)
        attention_weights = paddle.reshape(attention_weights, (1, 128, 1))

        # Weighted output
        weighted_output = paddle.multiply(pooled_features, attention_weights)
        weighted_output = paddle.sum(weighted_output, axis=1)

        # Fully connected layers
        fc1_out = paddle.relu(self.fc1(weighted_output))
        final_output = self.fc2(fc1_out)

        return final_output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Example input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print("Input tensor shape:", input_tensor.shape)
    print("Output tensor shape:", output_tensor.shape)