import paddle
import paddle.nn as nn
import paddle.io
import paddle.optimizer as optim

class DL_Model(paddle.nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.concat = nn.Concat(dim='mask')

        # Branch path
        self.conv_branch = nn.Conv2D(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0)

        # Fully connected layers
        self.fc1 = nn.FC(in_features=64, out_features=128, activation='relu')
        self.fc2 = nn.FC(in_features=128, out_features=10, activation=None)

    def forward(self, x):
        # Main path
        main_path = x
        main_path = self.conv1(main_path)
        main_path = self.conv2(main_path)
        main_path = self.conv3(main_path)
        main_path = self.concat(main_path)

        # Branch path
        branch_path = self.conv_branch(x)

        # Add main and branch paths
        output = self.add(main_path, branch_path)

        # Fully connected layers
        output = self.fc1(output)
        output = self.fc2(output)

        return output

    def add(self, x, y):
        return paddle.add(x, y)


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = paddle.randn(batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)