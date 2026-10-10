import sys
import paddle
import paddle.nn as nn

class DL_Model(paddle.nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.depthwise_conv1 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Block 2
        self.conv_block1 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(num_features=64)
        self.relu1 = nn.ReLU()
        self.reshaper = paddle.nn.ReshapeTransform(in_event_shape=(32, 32, 4, 64), out_event_shape=(32, 32, 4, 64))
        self.permute = nn.Identity()
        self.reshape_back = paddle.nn.ReshapeTransform(in_event_shape=(32, 32, 4, 64), out_event_shape=(32, 32, 32, 64))
        self.shuffle = nn.Identity()
        self.fc = nn.Linear(in_features=64 * 32 * 32, out_features=10)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.depthwise_conv1(x)
        x = self.conv2(x)

        # Block 2
        x = self.conv_block1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.reshaper(x)
        x = self.permute(x)
        x = self.reshape_back(x)
        x = self.shuffle(x)

        # Fully connected layer
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 1, 28, 28))  # Example input shape
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)