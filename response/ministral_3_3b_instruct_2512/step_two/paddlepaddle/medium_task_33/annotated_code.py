import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Split the input image into three channel groups
        self.conv1x1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)

        # Apply separable convolutions to each group
        self.sep_conv1x1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.sep_conv3x3 = nn.Conv2D(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')
        self.sep_conv5x5 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)

        # Apply LeakyReLU activation
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

        # Concatenate the outputs from the three groups
        self.concat = nn.AdaptiveAvgPool2D(output_size=(1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split the input image into three channel groups
        conv1x1_out = self.conv1x1(x)  # output tensor shape: (1, 64, 32, 32)
        conv3x3_out = self.conv3x3(x)  # output tensor shape: (1, 64, 32, 32)
        conv5x5_out = self.conv5x5(x)  # output tensor shape: (1, 64, 32, 32)

        # Apply separable convolutions to each group
        sep_conv1x1_out = self.sep_conv1x1(self.leaky_relu(conv1x1_out))  # output tensor shape: (1, 64, 32, 32)
        sep_conv3x3_out = self.sep_conv3x3(self.leaky_3x3_out(conv3x3_out))
        sep_conv5x5_out = self.sep_conv5x5(self.leaky_relu(conv5x5_out))

        # Concatenate the outputs
        concatenated = paddle.concat([sep_conv1x1_out, sep_conv3x3_out, sep_conv5x5_out], axis=1)

        # Adaptive average pooling
        pooled = self.concat(concatenated)

        # Flatten the output
        flattened = paddle.flatten(pooled)

        # Fully connected layers
        fc1_out = self.fc1(flattened)
        fc2_out = self.fc2(self.leaky_relu(fc1_out))
        fc3_out = self.fc3(self.neural_activation(fc2_out))

        return fc3_out

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)