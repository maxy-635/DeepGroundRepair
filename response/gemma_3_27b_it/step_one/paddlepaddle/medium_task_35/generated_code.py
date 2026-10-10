import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()

        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5 = nn.Conv2D(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2D(in_channels=3, out_channels=10, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.dropout1(x)

        x = self.upsample1(x)
        x = self.conv4(x)
        x = self.relu4(x)

        x = self.upsample2(x)
        x = self.conv5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)