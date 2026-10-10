import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.block = self.create_block()
        self.branch1 = self.create_branch('mask')
        self.branch2 = self.create_branch(self.block)
        self.concat = nn.Concat(axis=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32 * 32 * 3, out_features=10)
        self.softmax = nn.Softmax(axis=1)

    def create_block(self):
        block = nn.Sequential(
            nn.AvgPool2D(kernel_size=8, stride=8, padding=0),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=3),
        )
        return block

    def create_branch(self, block):
        branch = nn.Sequential(
            block,
            lambda x: paddle.reshape_(x, [x.shape[0], 3, x.shape[2], x.shape[3]]),
            lambda x: paddle.multiply(x, x, out=x),
        )
        return branch

    def forward(self, x):
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        concat_output = self.concat([branch1_output, branch2_output])
        flatten_output = self.flatten(concat_output)
        linear_output = self.linear(flatten_output)
        softmax_output = self.softmax(linear_output)
        return softmax_output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)