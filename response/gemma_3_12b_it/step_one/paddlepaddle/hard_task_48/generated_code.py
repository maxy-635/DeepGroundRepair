import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_group1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=64)
        )
        self.block1_group2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=64)
        )
        self.block1_group3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2D(num_features=64)
        )

        # Block 2 - Path 1
        self.path1 = nn.Sequential(
            nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Block 2 - Path 2
        self.path2 = nn.Sequential(
            nn.AvgPool2D(kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Block 2 - Path 3
        self.path3_conv1 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path3_conv1_1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path3_conv1_3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Block 2 - Path 4
        self.path4_conv1 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path4_conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.path4_conv3_1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path4_conv3_3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=64 * 4, out_features=10)

    def forward(self, x):
        # Block 1
        group1 = self.block1_group1(x)
        group2 = self.block1_group2(x)
        group3 = self.block1_group3(x)
        merged = paddle.concat([group1, group2, group3], axis=1)

        # Block 2
        path1 = self.path1(merged)
        path2 = self.path2(merged)
        path3_conv1 = self.path3_conv1(merged)
        path3_conv1_1 = self.path3_conv1_1(path3_conv1)
        path3_conv1_3 = self.path3_conv1_3(path3_conv1)
        path3 = paddle.concat([path3_conv1_1, path3_conv1_3], axis=1)

        path4_conv1 = self.path4_conv1(merged)
        path4_conv2 = self.path4_conv2(path4_conv1)
        path4_conv3_1 = self.path4_conv3_1(path4_conv2)
        path4_conv3_3 = self.path4_conv3_3(path4_conv2)
        path4 = paddle.concat([path4_conv3_1, path4_conv3_3], axis=1)

        merged_paths = paddle.concat([path1, path2, path3, path4], axis=1)
        
        flattened = self.flatten(merged_paths)
        output = self.linear(flattened)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)