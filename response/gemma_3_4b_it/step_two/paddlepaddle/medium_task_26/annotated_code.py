import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Compression layer
        self.comp_conv = nn.Conv2D(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')

        # Expansion layers
        self.exp_conv1 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.exp_conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Concatenation layer
        self.concat_axis = 1

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # Assuming 10 classes

    def forward(self, x):
        # the initial shape of x is: (32, 32, 64)
        # Compression
        x = self.comp_conv(x)

        # Expansion
        x = self.exp_conv1(x)
        x = self.exp_conv2(x)

        # Concatenation
        x = paddle.concat([x, x], axis=self.concat_axis)

        # Flatten
        x = self.flatten(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(32, 32, 64))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)