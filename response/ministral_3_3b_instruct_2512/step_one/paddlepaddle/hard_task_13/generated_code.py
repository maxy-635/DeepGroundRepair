import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: four parallel branches
        self.conv1x1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.max_pool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        # Second block: global average pooling and fully connected layers
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.fc1 = nn.Linear(in_features=64, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=3)

        # Activation functions
        self.relu = nn.ReLU()
        self.gelu = nn.GELU(approximate=True)

    def forward(self, x):
        # First block: parallel branches
        conv1x1_out = self.conv1x1(x)
        conv3x3_out = self.conv3x3(x)
        conv5x5_out = self.conv5x5(x)
        max_pool_out = self.max_pool(x)

        # Concatenate outputs
        concatenated = paddle.concat([conv1x1_out, conv3x3_out, conv5x5_out, max_pool_out], axis=1)

        # Second block: global average pooling and fully connected layers
        pooled_features = self.avg_pool(concatenated)
        pooled_features = paddle.flatten(pooled_features, start_dim=1)

        # Fully connected layers
        fc1_out = self.fc1(pooled_features)
        fc2_out = self.fc2(fc1_out)

        return fc2_out

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input tensor shape:", input_tensor.shape)
    print("Output tensor shape:", output_tensor.shape)