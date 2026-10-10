import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Stage 1: Convolution and Max Pooling
        self.conv1_1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        self.conv1_2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Stage 2: Convolution and Dropout
        self.conv2_1 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.25)

        self.conv2_2 = nn.Conv2D(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.25)

        # Stage 3: Upsampling and Skip Connections
        self.conv_up1 = nn.Conv2DTranspose(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv_skip1_1 = nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()

        self.conv_up2 = nn.Conv2DTranspose(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv_skip2_1 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()

        # Output Layer
        self.conv_out = nn.Conv2D(in_channels=128, out_channels=10, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(axis=1)


    def forward(self, x):
        # Stage 1
        x = self.pool1(self.relu1(self.conv1_1(x)))
        x = self.pool2(self.relu2(self.conv1_2(x)))

        # Stage 2
        x = self.relu3(self.conv2_1(x))
        x = self.dropout1(x)
        x = self.relu4(self.conv2_2(x))
        x = self.dropout2(x)

        # Stage 3
        x = self.conv_up1(x)
        x = F.interpolate(x, size=(x.shape[2] * 2, x.shape[3] * 2), mode='nearest')
        x = self.relu5(self.conv_skip1_1(x))

        x = self.conv_up2(x)
        x = F.interpolate(x, size=(x.shape[2] * 2, x.shape[3] * 2), mode='nearest')
        x = self.relu6(self.conv_skip2_1(x))

        x = self.conv_out(x)
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)