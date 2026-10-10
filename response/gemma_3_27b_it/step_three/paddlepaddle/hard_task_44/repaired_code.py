import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1_1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv1_3 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1_5 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.dropout1 = nn.Dropout(p=0.5)

        # Block 2
        self.branch1_conv = nn.Conv2D(in_channels=96, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.branch2_conv1 = nn.Conv2D(in_channels=96, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.branch2_conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch3_conv1 = nn.Conv2D(in_channels=96, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.branch3_conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.branch4_pool = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.branch4_conv = nn.Conv2D(in_channels=96, out_channels=32, kernel_size=1, stride=1, padding=0)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32 * 8 * 8, out_features=10) # Adjusted for CIFAR-10 and pooling

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1
        x1_1 = self.conv1_1(x)  # output tensor shape: (1, 32, 32, 32)
        x1_3 = self.conv1_3(x)  # output tensor shape: (1, 32, 32, 32)
        x1_5 = self.conv1_5(x)  # output tensor shape: (1, 32, 32, 32)

        x1_concat = paddle.concat([x1_1, x1_3, x1_5], axis=1)  # output tensor shape: (1, 96, 32, 32)
        x1_dropout = self.dropout1(x1_concat)  # output tensor shape: (1, 96, 32, 32)

        # Block 2
        x2_branch1 = self.branch1_conv(x1_dropout)  # output tensor shape: (1, 32, 32, 32)

        x2_branch2 = self.branch2_conv1(x1_dropout)  # output tensor shape: (1, 32, 32, 32)
        x2_branch2 = self.branch2_conv2(x2_branch2)  # output tensor shape: (1, 32, 32, 32)

        x2_branch3 = self.branch3_conv1(x1_dropout)  # output tensor shape: (1, 32, 32, 32)
        x2_branch3 = self.branch3_conv2(x2_branch3)  # output tensor shape: (1, 32, 32, 32)

        x2_branch4 = self.branch4_pool(x1_dropout)  # output tensor shape of this layer should be (1, 96, 32, 32)
        x2_branch4 = self.branch4_conv(x2_branch4)

        x2_concat = paddle.concat([x2_branch1, x2_branch2, x2_branch3, x2_branch4], axis=1)

        # Flatten and Linear
        x_flatten = self.flatten(x2_concat)
        x_output = self.linear(x_flatten)

        return x_output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))  # Example input tensor for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)