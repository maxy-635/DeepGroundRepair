import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Average pooling layer with 5x5 window and 3x3 stride
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))

        # 1x1 convolutional layer
        self.conv1x1 = nn.Conv2D(
            in_channels=1,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Flatten layer
        self.flatten = nn.Flatten()

        # First fully connected layer
        self.fc1 = nn.Linear(in_features='mask', out_features=128)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

        # Second fully connected layer
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Average pooling with 5x5 window and 3x3 stride
        # Note: AdaptiveAvgPool2D doesn't support stride directly, so we'll use a custom pooling layer
        # For the actual task, we'll use a regular pooling layer with the specified parameters
        # Here's the corrected implementation:
        self.pool = nn.AvgPool2D(kernel_size=5, stride=3, padding=0)
        x = self.pool(x)  # output tensor shape: (1, 1, 8, 8)

        # 1x1 convolutional layer
        x = self.conv1x1(x)  # output tensor shape: (1, 32, 8, 8)

        # Flatten layer
        x = self.flatten(x)  # output tensor shape: (1, 2048)

        # First fully connected layer
        x = self.fc1(x)

        # Dropout layer
        x = self.dropout(x)

        # Second fully connected layer
        x = self.fc2(x)

        # Softmax activation
        x = F.softmax(x, axis=1)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, channels, height, width]
    input_tensor = paddle.randn(shape=(1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)