import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.fc1 = nn.Linear(in_features=1, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x = self.adaptive_avg_pool(x)  # output tensor shape: (1, 3, 1, 1)
        weights = self.fc1(x)  # output tensor shape: (1, 3, 1, 10)
        weights = paddle.reshape(x, shape=[1, 3, 1, 10])
        x = x.expand_as(weights)
        x = x * weights
        x = self.flatten(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.rand(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)