import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main pathway: 1x1 convolution
        self.conv1x1_main = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )
        # Parallel branch: 1x1, 1x3, and 3x1 convolutions
        self.conv1x1_branch = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.conv1x3_branch = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv3x1_branch = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        # Concatenation and final 1x1 convolution
        self.conv1x1_output = nn.Conv2D(
            in_channels=64 * 3,  # Concatenation of 3 paths
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )
        # Direct connection from input
        self.direct_connection = nn.Conv2D(
            in_channels=3,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )
        # Fully connected layers for classification
        self.fc1 = nn.Linear(
            in_features=64 * 32 * 32,
            out_features=128
        )
        self.fc2 = nn.Linear(
            in_features=128,
            out_features=10
        )

    def forward(self, x):
        # Main pathway
        main_path = self.conv1x1_main(x)
        main_path = F.relu(main_path)

        # Parallel branch
        branch_1x1 = self.conv1x1_branch(x)
        branch_1x1 = F.relu(branch_1x1)
        branch_1x3 = self.conv1x3_branch(x)
        branch_1x3 = F.relu(branch_1x3)
        branch_3x1 = self.conv3x1_branch(x)
        branch_3x1 = F.relu(branch_3x1)

        # Concatenate the outputs of the parallel branch
        branch_output = paddle.concat([branch_1x1, branch_1x3, branch_3x1], axis=1)

        # Concatenate main path and branch output
        combined_output = paddle.concat([main_path, branch_output], axis=1)

        # Final 1x1 convolution
        output = self.conv1x1_output(combined_output)
        output = F.relu(output)

        # Additive connection with direct input
        output = output + self.direct_connection(x)
        output = F.relu(output)

        # Flatten for fully connected layers
        output = output.reshape([output.shape[0], -1])

        # Fully connected layers
        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (CIFAR-10 image size: 32x32x3)
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # Batch size of 1
    output_tensor = model(input_tensor)
    print(output_tensor.shape)