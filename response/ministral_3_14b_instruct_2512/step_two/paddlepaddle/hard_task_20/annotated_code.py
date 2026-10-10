import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path: three groups with different kernel sizes
        self.conv1x1_main = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.conv3x3_main = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv5x5_main = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

        # Branch path: 1x1 convolution
        self.conv1x1_branch = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels='mask', kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Concatenate the outputs from the three main path groups
        self.concat = nn.Sequential()

        # Fully connected layers for classification
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path processing
        x1x1 = self.conv1x1_main(x)  # output tensor shape: (1, 64, 32, 32)
        x3x3 = self.conv3x3_main(x)  # output tensor shape: (1, 64, 32, 32)
        x5x5 = self.conv5x5_main(x)  # output tensor shape: (1, 64, 32, 32)

        # Concatenate the outputs from the three main path groups
        concat_main = paddle.concat([x1x1, x3x3, x5x5], axis=1)  # output tensor shape: (1, 192, 32, 32)

        # Branch path processing
        x_branch = self.conv1x1_branch(x)  # output tensor shape of this layer should be (1, 192, 32, 32)

        # Combine main and branch paths through addition
        fused_features = concat_main + x_branch

        # Flatten the fused features for the fully connected layers
        flattened = paddle.flatten(fused_features, start_axis=1)

        # Fully connected layers
        x = self.fc1(flattened)
        x = F.relu(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)