import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Three average pooling layers with different scales
        self.avg_pool1 = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.avg_pool2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.avg_pool3 = nn.AvgPool2D(kernel_size=4, stride=4, padding=0)

        # Fully connected layer and reshape operation
        self.fc1 = nn.Linear(in_features=784, out_features=1024)
        self.reshape = paddle.reshape

        # Second block: Depthwise separable convolutional layers
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, groups=1)
        self.conv2 = nn.Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, groups=1)
        self.conv3 = nn.Conv2D(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=0, groups=1)
        self.conv4 = nn.Conv2D(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=0, groups=1)

        # Final fully connected layer
        self.fc2 = nn.Linear(in_features=1024, out_features=10)

    def forward(self, x):
        # First block: Process through three average pooling layers
        x1 = self.avg_pool1(x)
        x2 = self.avg_pool2(x)
        x3 = self.avg_pool3(x)

        # Flatten and concatenate the outputs
        x1_flat = paddle.flatten(x1)
        x2_flat = paddle.flatten(x2)
        x3_flat = paddle.flatten(x3)
        x_concat = paddle.concat([x1_flat, x2_flat, x3_flat], axis=1)

        # Transform the output of the first block into a 4-dimensional tensor
        x_fc = self.fc1(x_concat)
        x_reshape = self.reshape(x_fc, shape=[-1, 1, 32, 32])

        # Second block: Split and process through depthwise separable convolutional layers
        x_split = paddle.split(x_reshape, num_or_sections=4, axis=-1)
        x_conv1 = self.conv1(x_split[0])
        x_conv2 = self.conv2(x_split[1])
        x_conv3 = self.conv3(x_split[2])
        x_conv4 = self.conv4(x_split[3])

        # Concatenate the outputs
        x_conv_concat = paddle.concat([x_conv1, x_conv2, x_conv3, x_conv4], axis=-1)

        # Flatten and pass through the final fully connected layer
        x_flat = paddle.flatten(x_conv_concat)
        x_out = self.fc2(x_flat)

        return x_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)