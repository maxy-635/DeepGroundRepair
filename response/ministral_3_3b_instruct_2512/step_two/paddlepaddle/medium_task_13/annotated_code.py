import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=32)
        self.layer_norm2 = nn.LayerNorm(normalized_shape='mask')
        self.layer_norm3 = nn.LayerNorm(normalized_shape=128)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x = self.conv1(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.layer_norm1(x)  # output tensor shape: (1, 32, 32, 32)
        x = paddle.nn.ReLU()(x)  # output tensor shape: (1, 32, 32, 32)

        x = paddle.concat([x, self.conv2(x)], axis=1)  # output tensor shape: (1, 96, 32, 32)
        x = self.layer_norm2(x)
        x = paddle.nn.ReLU()(x)

        x = paddle.concat([x, self.conv3(x)], axis=1)
        x = self.layer_norm3(x)
        x = paddle.nn.ReLU()(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = paddle.nn.ReLU()(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Example input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)