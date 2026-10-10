import paddle
import paddle.nn as nn
import paddle.vision.datasets as datasets
import paddle.io

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        self.conv1 = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(
            in_channels=64,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.relu2 = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(64 * 1 * 1, 512)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(512, 10)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.avgpool(x)
        x = self.flatten(x)

        branch1 = x
        branch2 = x

        branch1 = self.linear1(branch1)
        branch1 = self.relu3(branch1)
        branch2 = self.linear1(branch2)
        branch2 = self.relu3(branch2)

        output = branch1 + branch2
        output = self.linear2(output)
        output = self.softmax(output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)