import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main pathway - 1x1 convolution
        self.conv1x1_main = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Parallel branch - 1x1, 1x3, 3x1 convolutions
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
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1)
        )
        self.conv3x1_branch = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0)
        )

        # Concatenation and final 1x1 convolution
        self.conv_after_cat = nn.Conv2D(
            in_channels=192,  # 64 (main) + 64*3 (branch)
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Direct connection (identity)
        self.identity = nn.Identity()

        # Classification head
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        # Main pathway
        main_out = self.conv1x1_main(x)
        main_out = self.relu(main_out)

        # Parallel branch
        branch1 = self.conv1x1_branch(x)
        branch2 = self.conv1x3_branch(x)
        branch3 = self.conv3x1_branch(x)
        branch_out = paddle.concat([branch1, branch2, branch3], axis=1)
        branch_out = self.relu(branch_out)

        # Concatenate and process
        combined = paddle.concat([main_out, branch_out], axis=1)
        combined = self.conv_after_cat(combined)
        combined = self.relu(combined)

        # Add identity connection
        out = combined + self.identity(x)

        # Flatten for classification
        out = paddle.flatten(out, start_axis=1)

        # Classification head
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)