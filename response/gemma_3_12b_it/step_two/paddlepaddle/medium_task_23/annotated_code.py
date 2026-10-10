import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()

        # Path 1: Single 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Path 2: 1x1 -> 1x7 -> 7x1 convolutions
        self.path2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3)
        )

        # Path 3: 1x1 -> (1x7 + 7x1) x 2 convolutions
        self.path3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3)
        )

        # Path 4: Average pooling -> 1x1 convolution
        self.path4 = nn.Sequential(
            F.avg_pool2d(x=None, kernel_size=2, stride=2, padding=0),
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Fully connected layer
        self.fc = nn.Linear(in_features=64 * 32 * 32, out_features=10)  # Assuming input size 32x32

    def forward(self, x):
        # Pass the input through each path
        path1_out = self.path1(x)
        path2_out = self.path2(x)
        path3_out = self.path3(x)
        path4_out = self.path4(x)

        # Concatenate the outputs of all paths
        concatenated = paddle.concat([path1_out, path2_out, path3_out, path4_out], axis=1)

        # Flatten the concatenated tensor
        flattened = nn.Flatten()(concatenated)

        # Pass the flattened tensor through the fully connected layer
        output = self.fc(flattened)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (CIFAR-10 has shape 32x32x3)
    input_tensor = paddle.rand(shape=[1, 3, 32, 32])  # Batch size of 1
    output_tensor = model(input_tensor)
    print(output_tensor.shape)