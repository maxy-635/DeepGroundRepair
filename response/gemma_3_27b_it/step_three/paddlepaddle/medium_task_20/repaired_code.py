import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: 1x1 convolution
        self.path1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        # Path 2: 1x1 conv -> 3x3 conv -> 3x3 conv
        self.path2_conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.path2_conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.path2_conv3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # Path 3: 1x1 conv -> 3x3 conv
        self.path3_conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.path3_conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        # Path 4: MaxPool -> 1x1 conv
        self.path4_pool = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.path4_conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.relu4 = nn.ReLU()

        # Concatenation
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32 * 32 * 4, out_features=128)
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Forward pass for each path
        path1_out = self.relu1(self.path1(x))  # output tensor shape: (1, 32, 32, 32)
        path2_out = self.relu2(self.path2_conv3(self.path2_conv2(self.path2_conv1(x))))  # output tensor shape: (1, 32, 32, 32)
        path3_out = self.relu3(self.path3_conv2(self.path3_conv1(x)))  # output tensor shape: (1, 32, 32, 32)
        path4_out = self.relu4(self.path4_conv1(self.path4_pool(x)))  # output tensor shape of this layer should be (1, 32, 32, 32)

        # Concatenate the outputs
        concat_out = paddle.concat([path1_out, path2_out, path3_out, path4_out], axis=1)

        # Flatten and pass through linear layer
        flattened_out = self.flatten(concat_out)
        linear_out = self.linear(flattened_out)

        # Softmax output
        output = self.softmax(linear_out)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)