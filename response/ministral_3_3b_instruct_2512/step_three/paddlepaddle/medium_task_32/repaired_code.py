import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Split the input into three groups along the last dimension
        self.conv1x1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)

        # Apply depthwise separable convolutional layers
        self.depthwise_conv1x1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.depthwise_conv3x3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.depthwise_conv5x5 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)

        # Apply LeakyReLU activation
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

        # Concatenate the outputs of the three groups
        self.concat = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.flatten = nn.Flatten()

        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # Split the input into three groups along the last dimension
        x1 = x[:, :, :, :1]  # First group
        x2 = x[:, :, :, 1:4]  # Second group
        x3 = x[:, :, :, 4:]  # Third group

        # Apply depthwise separable convolutional layers
        out1 = self.depthwise_conv1x1(self.leaky_relu(self.conv1x1(x1)))
        out2 = self.depthwise_conv3x3(self.leaky_relu(self.conv3x3(x2)))
        out3 = self.depthwise_conv5x5(self.leaky_relu(self.conv5x5(x3)))

        # Concatenate the outputs
        out = paddle.concat([out1, out2, out3], axis=1)

        # Flatten and pass through fully connected layer
        out = self.flatten(out)
        out = self.fc(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Dummy input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)