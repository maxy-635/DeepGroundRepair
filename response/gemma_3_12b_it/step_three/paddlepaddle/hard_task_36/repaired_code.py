import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()
        # Main Pathway
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.max_pool = nn.MaxPool2D(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)

        # Branch Pathway
        self.conv_branch = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Fusion and Classification
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        # Main Pathway
        x_main = self.conv1(x)
        x_main = self.conv2_1(x_main)
        x_main = self.conv2_2(x_main)
        x_main = self.max_pool(x_main)
        x_main = self.dropout(x_main)

        # Branch Pathway
        x_branch = self.conv_branch(x)

        # Fusion
        x_fuse = paddle.add(x_main, x_branch)

        # Classification
        x_pool = self.adaptive_avg_pool(x_fuse)
        x_flatten = self.flatten(x_pool)
        x_output = self.linear(x_flatten)

        return x_output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)