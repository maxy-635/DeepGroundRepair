import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.avgpool1 = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.avgpool2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.avgpool3 = nn.AvgPool2D(kernel_size=4, stride=4, padding=0)
        self.flatten1 = nn.Flatten()
        self.dropout1 = nn.Dropout(p=0.5)
        self.concat1 = nn.Concat(axis=1)
        self.fc1 = nn.Linear(in_features=1024, out_features=1024)
        self.reshape = nn.Reshape((1024, 1, 1, 1))

        # Block 2
        self.conv1_1 = nn.Conv2D(in_channels=1024, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv1_3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.avgpool_block2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.conv3_1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.concat2 = nn.Concat()
        self.fc2 = nn.Linear(in_features=64 * 4, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Block 1
        x1 = self.avgpool1(x)
        x2 = self.avgpool2(x)
        x3 = self.avgpool3(x)

        x1 = self.flatten1(x1)
        x2 = self.flatten1(x2)
        x3 = self.flatten1(x3)

        x1 = self.dropout1(x1)
        x2 = self.dropout1(x2)
        x3 = self.dropout1(x3)

        x = self.concat1([x1, x2, x3])
        x = self.fc1(x)
        x = self.reshape(x)

        # Block 2
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)

        x2 = self.conv2_1(x)
        x2 = self.conv2_2(x2)
        x2 = self.conv2_3(x2)

        x3 = self.avgpool_block2(x)
        x3 = self.conv3_1(x3)

        x = self.concat2([x1, x2, x3])
        x = self.fc2(x)
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)