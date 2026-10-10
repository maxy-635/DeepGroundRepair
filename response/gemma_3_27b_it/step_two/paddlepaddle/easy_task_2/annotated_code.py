import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.feature_extraction1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        )
        self.feature_extraction2 = nn.Sequential(
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        )
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features='mask', out_features=512)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=1000)
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 224, 224)
        x = self.feature_extraction1(x)  # output tensor shape: (1, 32, 112, 112)
        x = self.feature_extraction2(x)  # output tensor shape: (1, 64, 56, 56)
        x = self.conv3(x)  # output tensor shape: (1, 128, 56, 56)
        x = self.conv4(x)  # output tensor shape: (1, 256, 56, 56)
        x = self.conv5(x)  # output tensor shape: (1, 256, 56, 56)
        x = self.avgpool(x)  # output tensor shape: (1, 256, 28, 28)
        x = self.flatten(x)  # output tensor shape: (1, 200704)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 224, 224])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)