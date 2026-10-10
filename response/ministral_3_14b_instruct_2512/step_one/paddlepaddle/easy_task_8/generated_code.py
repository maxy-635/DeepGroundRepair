import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.depthwise_conv = nn.Conv2D(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias_attr=False
        )
        self.pointwise_conv = nn.Conv2D(
            in_channels=1,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=False
        )
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=32 * 28 * 28, out_features=10)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.pointwise_conv(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 1, 28, 28))  # batch_size=1, channels=1, height=28, width=28
    output_tensor = model(input_tensor)
    print(output_tensor.shape)