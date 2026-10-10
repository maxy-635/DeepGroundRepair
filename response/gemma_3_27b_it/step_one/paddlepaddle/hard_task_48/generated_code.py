import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_group1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(num_channels=32),
            nn.ReLU()
        )
        self.block1_group2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(num_channels=32),
            nn.ReLU()
        )
        self.block1_group3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm(num_channels=32),
            nn.ReLU()
        )

        # Block 2
        self.block2_path1 = nn.Sequential(
            nn.Conv2D(in_channels=96, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(num_channels=32),
            nn.ReLU()
        )
        self.block2_path2 = nn.Sequential(
            nn.AvgPool2D(kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(num_channels=32),
            nn.ReLU()
        )
        self.block2_path3 = nn.Sequential(
            nn.Conv2D(in_channels=96, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(num_channels=32),
            nn.ReLU()
        )
        self.block2_path3_subpath1 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=1, padding=0)
        self.block2_path3_subpath2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=(3, 1), stride=1, padding=0)
        self.block2_path4 = nn.Sequential(
            nn.Conv2D(in_channels=96, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm(num_channels=32),
            nn.ReLU()
        )
        self.block2_path4_conv = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.block2_path4_subpath1 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=1, padding=0)
        self.block2_path4_subpath2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=(3, 1), stride=1, padding=0)

        # Final layers
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32 * 32 * 32, out_features=10) # Adjust input features based on image size and block 2 output

    def forward(self, x):
        # Block 1
        x1 = self.block1_group1(x)
        x2 = self.block1_group2(x)
        x3 = self.block1_group3(x)
        x = paddle.concat([x1, x2, x3], axis=1)  # Concatenate along the channel dimension

        # Block 2
        path1 = self.block2_path1(x)
        path2 = self.block2_path2(x)
        path3 = self.block2_path3(x)
        path3_subpath1 = self.block2_path3_subpath1(path3)
        path3_subpath2 = self.block2_path3_subpath2(path3)
        path3 = paddle.concat([path3_subpath1, path3_subpath2], axis=1)

        path4 = self.block2_path4(x)
        path4_conv = self.block2_path4_conv(path4)
        path4_subpath1 = self.block2_path4_subpath1(path4_conv)
        path4_subpath2 = self.block2_path4_subpath2(path4_conv)
        path4 = paddle.concat([path4_subpath1, path4_subpath2], axis=1)

        x = paddle.concat([path1, path2, path3, path4], axis=1)

        # Final layers
        x = self.flatten(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Example input tensor for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)