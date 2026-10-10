import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path_block1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, groups=1, bias_attr=True),
        )
        self.main_path_block2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2D(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask', groups='mask', bias_attr='mask'),
        )
        self.main_path_block3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, groups=1, bias_attr=True),
        )

        # Branch Path
        self.branch_path_conv = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, groups=1, bias_attr=True)

        # Fully Connected Layer
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=64 * 28 * 28, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Main Path
        x_main = self.main_path_block1(x)  # output tensor shape: (1, 16, 28, 28)
        x_main = paddle.concat([x, x_main], axis=1)  # output tensor shape: (1, 17, 28, 28)
        x_main = self.main_path_block2(x_main)
        x_main = paddle.concat([x_main, x_main], axis=1)
        x_main = self.main_path_block3(x_main)
        x_main = paddle.concat([x_main, x_main], axis=1)

        # Branch Path
        x_branch = self.branch_path_conv(x)

        # Feature Fusion
        x_fused = paddle.add(x_main, x_branch)

        # Classification
        x_flattened = self.flatten(x_fused)
        output = self.linear(x_flattened)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)