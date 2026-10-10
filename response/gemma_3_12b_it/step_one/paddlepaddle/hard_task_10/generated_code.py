import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Feature extraction path 1: 1x1 convolution
        self.conv1x1_path1 = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.relu1 = nn.ReLU()

        # Feature extraction path 2: Sequence of convolutions
        self.conv1x1_path2 = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.conv1x7_path2 = nn.Conv2D(
            in_channels=64,
            out_channels=64,
            kernel_size=7,
            stride=1,
            padding=3
        )
        self.conv7x1_path2 = nn.Conv2D(
            in_channels=64,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.relu2 = nn.ReLU()

        # Concatenate the outputs of the two paths and apply a 1x1 convolution
        self.concat_conv = nn.Conv2D(
            in_channels=128,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.relu3 = nn.ReLU()

        # Branch connecting directly to the input
        self.branch_conv = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.relu4 = nn.ReLU()

        # Merge the outputs of the main path and the branch through addition
        # self.add_layer = nn.Add()  # Not required as paddle.add() handles addition

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(
            in_features=64 * 32 * 32,  # Assuming input size of 32x32
            out_features=128
        )
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(
            in_features=128,
            out_features=10
        )
        self.softmax = nn.Softmax(axis=-1)


    def forward(self, x):
        # Feature extraction path 1
        path1 = self.relu1(self.conv1x1_path1(x))

        # Feature extraction path 2
        path2 = self.relu2(self.conv7x1_path2(self.conv1x7_path2(self.conv1x1_path2(x)))))

        # Concatenate the outputs of the two paths
        merged_path = paddle.concat([path1, path2], axis=1)

        # Apply a 1x1 convolution to align the output dimensions
        main_path = self.relu3(self.concat_conv(merged_path))

        # Branch connecting directly to the input
        branch = self.relu4(self.branch_conv(x))

        # Merge the outputs of the main path and the branch
        combined = main_path + branch  # paddle.add(main_path, branch)

        # Classification results through fully connected layers
        flattened = self.flatten(combined)
        fc1_output = self.fc1(flattened)
        dropout_output = self.dropout(fc1_output)
        output = self.softmax(self.fc2(dropout_output))

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)