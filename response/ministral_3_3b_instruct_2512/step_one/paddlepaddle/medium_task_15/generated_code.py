import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial feature extraction
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.LayerNorm(normalized_shape=64)
        self.relu1 = nn.ReLU()

        # Global average pooling and FC layers
        self.avg_pool1 = nn.AvgPool2D(kernel_size=8, stride=1, padding=0)
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=66)  # Adjusted to match initial feature dimensions

        # Reshape and weighted feature maps
        self.reshape = paddle.distribution.ReshapeTransform(
            in_event_shape=(64, 1, 1, 1), out_event_shape=(64, 3, 3, 3)
        )

        # Dimensionality reduction and downsampling
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.avg_pool2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)

        # Final classification layer
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # Initial feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Global average pooling and FC layers
        x = self.avg_pool1(x)
        x = self.flatten1(x)
        x = self.fc1(x)
        x = self.fc2(x)

        # Reshape and weighted feature maps
        x = self.reshape(x)
        x = paddle.matmul(x, self.conv2.weight)  # Weighted feature maps
        x = paddle.concat([x, self.conv2.bias], axis=0)  # Concatenate bias if needed (assuming bias is handled differently)

        # Dimensionality reduction and downsampling
        x = self.avg_pool2(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = paddle.randn((1, 3, 32, 32))  # CIFAR-10 image dimensions
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)