import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=7, stride=1, padding=3, groups=3, bias_attr=False)
        self.ln = nn.LayerNorm(normalized_shape=[3, 32, 32])
        self.fc1 = nn.Linear(in_features=3 * 32 * 32, out_features=3 * 32 * 32)
        self.fc2 = nn.Linear(in_features=3 * 32 * 32, out_features=3 * 32 * 32)
        self.fc3 = nn.Linear(in_features=3 * 32 * 32, out_features=10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.ln(x)
        x = self.relu(x)
        x = x.reshape([x.shape[0], -1])
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = paddle.add(x, x.reshape([x.shape[0], -1]))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)