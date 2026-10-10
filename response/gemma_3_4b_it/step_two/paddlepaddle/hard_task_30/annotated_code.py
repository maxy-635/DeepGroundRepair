import paddle
import paddle.nn as nn
import paddle.io
import paddle.static

class DL_Model(paddle.nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Dual-Path Structure
        self.conv1_main = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_main = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3_main = nn.Conv2D(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0)

        self.conv1_branch = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_branch = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.add = nn.Add()
        self.concat = nn.Concat(dim=1)

        # Block 2: Channel Splitting
        self.conv_1x1 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv_3x3 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)

        self.fc1 = paddle.static.nn.fc(in_features=64, size=10, activation=None, name="fc1")
        self.fc2 = paddle.static.nn.fc(in_features=10, size=10, activation=None, name="fc2")

    def forward(self, x):
        # Block 1
        main_path = self.conv1_main(x)
        main_path = self.conv2_main(main_path)
        main_path = self.conv3_main(main_path)

        branch_path = self.conv1_branch(x)
        branch_path = self.conv2_branch(branch_path)

        output1 = self.add(main_path, branch_path)
        output1 = self.concat(output1, [main_path, branch_path])

        # Block 2
        group1 = self.conv_1x1(output1)
        group2 = self.conv_3x3(output1)
        group3 = self.conv_5x5(output1)

        output2 = self.concat(group1, group2, group3)

        # Fully Connected Layers
        output = self.fc1(output2)
        output = self.fc2(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)