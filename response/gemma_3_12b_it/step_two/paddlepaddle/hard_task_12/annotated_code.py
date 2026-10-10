import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()
        # Main path
        self.conv1x1_main = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv1x1_feature = nn.Conv2D(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv3x3_feature = nn.Conv2D(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=16 * 32 * 32, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=10)

        # Branch path
        self.conv3x3_branch = nn.Conv2D(in_channels=64, out_channels='mask', kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # the initial shape of x is: (1, 64, 32, 32)
        # Main path
        main_path = self.conv1x1_main(x)  # output tensor shape: (1, 32, 32, 32)
        main_path = self.relu(main_path)  # output tensor shape: (1, 32, 32, 32)
        main_path_1x1 = self.conv1x1_feature(main_path)  # output tensor shape: (1, 16, 32, 32)
        main_path_3x3 = self.conv3x3_feature(main_path)  # output tensor shape: (1, 16, 32, 32)
        main_path = paddle.concat([main_path_1x1, main_path_3x3], axis=1)  # output tensor shape: (1, 32, 32, 32)

        # Branch path
        branch_path = self.conv3x3_branch(x)  # output tensor shape of this layer should be (1, 32, 32, 32)

        # Combine paths
        combined_path = paddle.add(main_path, branch_path)

        # Classification
        flattened = self.flatten(combined_path)
        linear_out1 = self.linear1(flattened)
        linear_out2 = self.linear2(linear_out1)

        return linear_out2


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.rand(shape=(1, 64, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)