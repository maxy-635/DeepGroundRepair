import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_conv1x1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.main_branch1_conv3x3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch Path
        self.branch_conv1x1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch1_avgpool = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.branch1_conv3x3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch1_convtranspose = nn.Conv2DTranspose(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)

        self.branch2_avgpool = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.branch2_conv3x3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch2_convtranspose = nn.Conv2DTranspose(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)

        self.concat_conv1x1 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Classification
        self.fc = nn.Linear(in_features=64 * 8 * 8, out_features=10)  # Assuming input size after conv layers is 8x8

    def forward(self, x):
        # Main Path
        main_path = self.main_conv1x1(x)
        main_branch1 = self.main_branch1_conv3x3(main_path)

        # Branch Path
        branch_path = self.branch_conv1x1(x)
        branch1_down = self.branch1_avgpool(branch_path)
        branch1_conv = self.branch1_conv3x3(branch1_down)
        branch1_up = self.branch1_convtranspose(branch1_conv)

        branch2_down = self.branch2_avgpool(branch_path)
        branch2_conv = self.branch2_conv3x3(branch2_down)
        branch2_up = self.branch2_convtranspose(branch2_conv)


        # Concatenate and Fuse
        concat_tensor = paddle.concat([main_branch1, branch1_up, branch2_up], axis=1)
        fused_output = self.concat_conv1x1(concat_tensor)

        # Classification
        flattened_output = paddle.flatten(fused_output, [1, 2, 3])
        output = self.fc(flattened_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)