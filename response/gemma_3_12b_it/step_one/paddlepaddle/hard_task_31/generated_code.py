import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First Block
        self.main_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2D(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        )
        self.branch_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        )

        # Second Block
        self.conv1x1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.dropout1x1 = nn.Dropout(p=0.2)
        self.conv3x3 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.dropout3x3 = nn.Dropout(p=0.2)
        self.conv5x5 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.dropout5x5 = nn.Dropout(p=0.2)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32 * 32 * 3, out_features=10)  # Assuming CIFAR-10

    def forward(self, x):
        # First Block
        main_path_output = self.main_path(x)
        branch_path_output = self.branch_path(x)
        added_output = paddle.add(main_path_output, branch_path_output)

        # Second Block
        output_1x1 = self.dropout1x1(self.conv1x1(added_output))
        output_3x3 = self.dropout3x3(self.conv3x3(added_output))
        output_5x5 = self.dropout5x5(self.conv5x5(added_output))

        concatenated_output = paddle.concat([output_1x1, output_3x3, output_5x5], axis=-1)

        # Flatten and Linear
        flattened_output = self.flatten(concatenated_output)
        output = self.linear(flattened_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)