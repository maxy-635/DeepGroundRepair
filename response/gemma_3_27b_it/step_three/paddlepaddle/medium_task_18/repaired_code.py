import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1_1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv1_3 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1_5 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=32 * 16 * 16, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        conv1_1_out = self.relu(self.conv1_1(x))  # output tensor shape: (1, 32, 32, 32)
        conv1_3_out = self.relu(self.conv1_3(x))  # output tensor shape: (1, 32, 32, 32)
        conv1_5_out = self.relu(self.conv1_5(x))  # output tensor shape: (1, 32, 32, 32)
        
        concat_out = paddle.concat([conv1_1_out, conv1_3_out, conv1_5_out], axis=1)  # output tensor shape: (1, 96, 32, 32)
        pool_out = self.pool(concat_out)  # output tensor shape: (1, 96, 16, 16)
        
        flatten_out = self.flatten(pool_out)  # output tensor shape: (1, 24576)
        linear1_out = self.relu(self.linear1(flatten_out))
        output = self.linear2(linear1_out)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)