import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First Block
        self.main_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
        )
        self.branch_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        )

        # Second Block
        self.conv1x1 = nn.DepthwiseConv2D(in_channels='mask', kernel_size='mask', stride='mask', padding='mask')
        self.conv3x3 = nn.DepthwiseConv2D(in_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.DepthwiseConv2D(in_channels=3, kernel_size=5, stride=1, padding=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=3 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # First Block
        main_path_output = self.main_path(x)
        branch_path_output = self.branch_path(x)
        combined_output = main_path_output + branch_path_output

        # Second Block
        group1 = self.conv1x1(combined_output)
        group2 = self.conv3x3(combined_output)
        group3 = self.conv5x5(combined_output)
        concatenated_output = paddle.concat([group1, group2, group3], axis=1)

        # Flatten and Fully Connected Layers
        flattened_output = nn.Flatten()(concatenated_output)
        fc1_output = self.fc1(flattened_output)
        output = self.fc2(fc1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.rand(shape=[1, 3, 32, 32])  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)