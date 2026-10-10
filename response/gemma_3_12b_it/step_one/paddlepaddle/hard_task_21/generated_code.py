import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.split = paddle.split
        self.concat = paddle.concat
        self.conv1x1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)

        # Branch path
        self.conv1x1_branch = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=32 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Main path
        x1, x2, x3 = self.split(x, num_or_sections=3, axis=1)
        main_path_1 = self.conv1x1(x1)
        main_path_2 = self.conv3x3(x2)
        main_path_3 = self.conv5x5(x3)
        main_path_output = self.concat([main_path_1, main_path_2, main_path_3], axis=1)

        # Branch path
        branch_path_output = self.conv1x1_branch(x)

        # Add the outputs from both paths
        added_output = paddle.add(main_path_output, branch_path_output)

        # Flatten the output for fully connected layers
        flattened_output = paddle.flatten(added_output, [1, 2, 3])

        # Fully connected layers
        x = self.fc1(flattened_output)
        x = paddle.nn.functional.relu(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.rand(shape=[1, 3, 32, 32])  # Example input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)