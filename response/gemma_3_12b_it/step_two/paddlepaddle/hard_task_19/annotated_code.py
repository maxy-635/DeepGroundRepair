import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()
        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.AdaptiveAvgPool2D(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features='mask', out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256)
        )

        # Final layers
        self.final_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256 * 4 * 4, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (32, 3, 32, 32)
        # Main path
        main_path_output = self.main_path(x)  # output tensor shape: (32, 256, 16, 16)

        # Branch path
        branch_path_output = self.branch_path(x)
        branch_path_output = paddle.reshape(branch_path_output, [x.shape[0], 256])

        # Add the outputs from both paths and pass through final layers
        added_output = main_path_output + branch_path_output
        final_output = self.final_layers(added_output)

        return final_output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for CIFAR-10 (batch_size, channels, height, width)
    input_tensor = paddle.randn(shape=[32, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)