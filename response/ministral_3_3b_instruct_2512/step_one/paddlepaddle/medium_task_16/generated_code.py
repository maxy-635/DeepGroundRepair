import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Split input into three groups and apply 1x1 convolutions
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Downsampling via average pooling
        self.pool = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)

        # Flatten and fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=3 * 16, out_features=128)  # Assuming input size after pooling is 16x16
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Split and process each group
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        # Apply pooling to each group
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)

        # Concatenate along channel dimension
        x = paddle.concat([x1, x2, x3], axis=1)

        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = paddle.nn.functional.relu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = paddle.randn((1, 3, 32, 32))  # CIFAR-10 images are 32x32
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)