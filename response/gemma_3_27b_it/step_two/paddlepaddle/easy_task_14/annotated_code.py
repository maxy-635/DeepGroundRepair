import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.avg_pool = nn.AvgPool2D(kernel_size=(8, 8), stride=(8, 8))
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(in_features=64 * 64, out_features=10)
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x = self.avg_pool(x)  # output tensor shape: (1, 3, 4, 4)
        x = self.flatten(x)  # output tensor shape: (1, 48)
        weights1 = self.relu(self.fc1(x))
        weights2 = self.relu(self.fc2(x))

        x = x * weights1 * weights2
        x = self.fc3(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)