import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.conv1x1 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3x3_1 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3x3_2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Branch path - ensure output channels match main path
        self.branch_conv = nn.Conv2D(in_channels=32, out_channels='mask', kernel_size=3, stride=1, padding=1)

        # Flatten and FC layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32*32*96, out_features=128)  # 32x32x96 after concatenation
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 32, 32, 64)
        # Main path
        x1 = self.conv1x1(x)  # output tensor shape: (1, 32, 32, 64)
        x2 = self.conv3x3_1(x)  # output tensor shape: (1, 32, 32, 64)
        x3 = self.conv3x3_2(x)  # output tensor shape: (1, 32, 32, 64)
        main_path_output = paddle.concat([x1, x2, x3], axis=1)  # output tensor shape: (1, 96, 32, 64)

        # Branch path
        branch_output = self.branch_conv(x)  # output tensor shape of this layer should be (1, 96, 32, 64)

        # Combine outputs
        combined_output = paddle.add(main_path_output, branch_output)

        # Flatten and FC layers
        flattened_output = self.flatten(combined_output)
        fc_output = self.fc1(flattened_output)
        fc_output = paddle.nn.functional.relu(fc_output)
        output = self.fc2(fc_output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 32, 32, 64))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)