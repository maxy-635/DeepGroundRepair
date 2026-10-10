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
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1)
        )
        self.conv3x1_branch = nn.Conv2D(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0)
        )

        # Concatenation and another 1x1 convolution
        self.conv1x1_fusion = nn.Conv2D(
            in_channels=128,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Direct connection (skip connection)
        self.skip_connection = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Fully connected layers for classification
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # Main pathway
        main_path = self.conv1x1_main(x)

        # Parallel branch
        branch = self.conv1x1_branch(x)
        branch = self.conv1x3_branch(branch)
        branch = self.conv3x1_branch(branch)

        # Concatenation and fusion
        concatenated = paddle.concat([main_path, branch], axis=1)
        fused = self.conv1x1_fusion(concatenated)

        # Skip connection
        skip = self.skip_connection(x)
        output = paddle.add(fused, skip)

        # Flatten and fully connected layers
        output = paddle.flatten(output, start_axis=1)
        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]