import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(num_features=32)
        self.relu1 = nn.ReLU()

        # Block 2
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(num_features=64)
        self.relu2 = nn.ReLU()

        # Block 3
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2D(num_features=128)
        self.relu3 = nn.ReLU()

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=128 * 32 * 32, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # Block 1
        out1 = self.relu1(self.bn1(self.conv1(x)))

        # Block 2 with concatenation
        out2 = self.relu2(self.bn2(self.conv2(out1)))
        out2 = paddle.concat([out1, out2], axis=1)

        # Block 3 with concatenation
        out3 = self.relu3(self.bn3(self.conv3(out2)))
        out3 = paddle.concat([out2, out3], axis=1)

        # Fully connected layers
        out = self.flatten(out3)
        out = self.fc1(out)
        out = self.fc2(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]