import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path_conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.main_path_branch1 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.main_path_branch2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.main_path_branch2_conv = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.main_path_branch2_deconv = nn.Conv2DTranspose(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.main_path_branch3 = nn.AvgPool2D(kernel_size=4, stride=4, padding=0)
        self.main_path_branch3_conv = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.main_path_branch3_deconv = nn.Conv2DTranspose(in_channels=32, out_channels=16, kernel_size=4, stride=4, padding=0)
        self.main_path_conv2 = nn.Conv2D(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Branch Path
        self.branch_path_conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)

        # Fully Connected Layer
        self.fc = nn.Linear(in_features=64 * 8 * 8, out_features=10)  # Assuming input size is 32x32

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main Path
        x_main = self.main_path_conv1(x)  # output tensor shape: (1, 16, 32, 32)
        branch1 = self.main_path_branch1(x_main)  # output tensor shape: (1, 32, 32, 32)
        branch2 = self.main_path_branch2(x_main)  # output tensor shape: (1, 16, 16, 16)
        branch2 = self.main_path_branch2_conv(branch2)  # output tensor shape: (1, 32, 16, 16)
        branch2 = self.main_path_branch2_deconv(branch2)  # output tensor shape: (1, 16, 32, 32)
        branch3 = self.main_path_branch3(x_main)  # output tensor shape: (1, 16, 8, 8)
        branch3 = self.main_path_branch3_conv(branch3)  # output tensor shape: (1, 32, 8, 8)
        branch3 = self.main_path_branch3_deconv(branch3)  # output tensor shape: (1, 16, 32, 32)
        x_main = paddle.concat([branch1, branch2, branch3], axis=1)  # output tensor shape: (1, 64, 32, 32)
        x_main = self.main_path_conv2(x_main)

        # Branch Path
        x_branch = self.branch_path_conv1(x)

        # Fusion
        x_out = x_main + x_branch

        # Fully Connected Layer
        x_out = paddle.reshape(x_out, [x_out.shape[0], 64 * 8 * 8])
        x_out = self.fc(x_out)

        return x_out


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)