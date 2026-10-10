import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(in_features=64*8*8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        pool_out = self.pool(conv2_out)
        residual = paddle.add(x, pool_out)
        flattened = paddle.flatten(residual, start_axis=1)
        fc1_out = self.fc1(flattened)
        output = self.fc2(fc1_out)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)