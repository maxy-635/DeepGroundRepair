import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Split the input image into three groups along the channel dimension
        # 1x1 convolutional kernel
        self.conv1x1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        # 3x3 convolutional kernel
        self.conv3x3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 5x5 convolutional kernel
        self.conv5x5 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)

        # Layer normalization for each convolutional group
        self.norm1x1 = nn.LayerNorm(normalized_shape=[64])
        self.norm3x3 = nn.LayerNorm(normalized_shape=[64])
        self.norm5x5 = nn.LayerNorm(normalized_shape=[64])

        # Leaky ReLU activation functions
        self.relu1x1 = nn.LeakyReLU(negative_slope=0.1)
        self.relu3x3 = nn.LeakyReLU(negative_slope=0.1)
        self.relu5x5 = nn.LeakyReLU(negative_slope=0.1)

        # Adaptive average pooling
        self.pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 1 * 1, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Process each convolutional group
        conv1x1_out = self.conv1x1(x)
        conv1x1_out = self.norm1x1(conv1x1_out)
        conv1x1_out = self.relu1x1(conv1x1_out)

        conv3x3_out = self.conv3x3(x)
        conv3x3_out = self.norm3x3(conv3x3_out)
        conv3x3_out = self.relu3x3(conv3x3_out)

        conv5x5_out = self.conv5x5(x)
        conv5x5_out = self.norm5x5(conv5x5_out)
        conv5x5_out = self.relu5x5(conv5x5_out)

        # Concatenate the outputs from the three convolutional groups
        fused_features = paddle.concat([conv1x1_out, conv3x3_out, conv5x5_out], axis=1)

        # Flatten and pass through fully connected layers
        pooled_features = self.pool(fused_features)
        flattened_features = self.flatten(pooled_features)

        # Apply dropout and fully connected layers
        x = self.dropout(flattened_features)
        x = self.fc1(x)
        x = self.relu1x1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = paddle.randn((1, 3, 32, 32))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input tensor shape:", input_tensor.shape)
    print("Output tensor shape:", output_tensor.shape)