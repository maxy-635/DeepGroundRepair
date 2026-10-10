import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()
        # Main path
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2D(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Branch path
        self.branch_conv = nn.Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Flatten and linear layers
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features='mask', out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Main path
        main_path = self.conv1(x)  # output tensor shape: (1, 16, 28, 28)
        main_path = self.relu1(main_path)  # output tensor shape: (1, 16, 28, 28)
        main_path = self.dropout1(main_path)  # output tensor shape: (1, 16, 28, 28)
        main_path = self.conv2(main_path)  # output tensor shape: (1, 32, 28, 28)
        main_path = self.relu2(main_path)  # output tensor shape: (1, 32, 28, 28)
        main_path = self.conv3(main_path)  # output tensor shape: (1, 16, 28, 28)

        # Branch path
        branch_path = self.branch_conv(x)  # output tensor shape: (1, 16, 28, 28)

        # Combine paths
        combined_path = paddle.add(main_path, branch_path)  # output tensor shape: (1, 16, 28, 28)

        # Flatten and linear layers
        flattened_output = self.flatten(combined_path)  # output tensor shape: (1, 12544)
        output = self.linear(flattened_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])  # Dummy input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)