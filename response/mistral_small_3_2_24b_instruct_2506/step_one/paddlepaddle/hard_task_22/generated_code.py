import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path components
        self.conv1x1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)

        # Branch path component
        self.branch_conv = nn.Conv2D(in_channels=3, out_channels=192, kernel_size=1, stride=1, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=192 * 8 * 8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

        # Flatten layer
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Split input into three groups along the channel dimension
        x_split = paddle.split(x, num_or_sections=3, axis=1)

        # Main path: multi-scale feature extraction
        out1 = self.conv1x1(x_split[0])
        out2 = self.conv3x3(x_split[1])
        out3 = self.conv5x5(x_split[2])

        # Concatenate the outputs from the main path
        main_out = paddle.concat([out1, out2, out3], axis=1)

        # Branch path: 1x1 convolution
        branch_out = self.branch_conv(x)

        # Fuse the outputs from both paths
        fused_out = main_out + branch_out

        # Flatten and pass through fully connected layers
        flattened = self.flatten(fused_out)
        fc_out = self.fc1(flattened)
        fc_out = F.relu(fc_out)
        output = self.fc2(fc_out)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # CIFAR-10 image shape: 3x32x32
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)