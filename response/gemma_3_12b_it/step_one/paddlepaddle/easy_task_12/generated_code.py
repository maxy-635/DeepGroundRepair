import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        )

        self.main_path_block2 = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2D(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Final layers
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=64 * 7 * 7, out_features=10)

    def forward(self, x):
        # Main path
        main_path_output = self.main_path(x)
        main_path_output = self.main_path_block2(main_path_output)

        # Branch path
        branch_path_output = self.branch_path(x)

        # Sum the outputs
        summed_output = main_path_output + branch_path_output

        # Flatten and linear
        flattened_output = self.flatten(summed_output)
        output = self.linear(flattened_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])  # Dummy input tensor (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)