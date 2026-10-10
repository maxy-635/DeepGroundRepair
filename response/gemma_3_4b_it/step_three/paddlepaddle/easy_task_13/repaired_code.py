import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.dropout1 = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(p=0.25)
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.dropout3 = nn.Dropout(p=0.25)
        self.conv4 = nn.Conv2D(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=128 * 128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.dropout3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 1, 28, 28), dtype=paddle.float32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)