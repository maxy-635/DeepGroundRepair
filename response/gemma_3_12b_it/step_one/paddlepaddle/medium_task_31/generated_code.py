import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the convolutional layers with different kernel sizes
        self.conv1x1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 5 * 5, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Apply different convolutional kernels to three groups
        out1 = self.conv1x1(x)
        out2 = self.conv3x3(x)
        out3 = self.conv5x5(x)

        # Concatenate the outputs
        fused_features = paddle.concat([out1, out2, out3], axis=1)

        # Flatten the features
        flattened_features = nn.Flatten()(fused_features)

        # Pass through fully connected layers
        x = self.fc1(flattened_features)
        x = paddle.nn.ReLU()(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor with shape [1, 3, 32, 32]
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]