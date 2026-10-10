import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self, in_channels=3, num_classes=10):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        # Branch 3: 1x1 convolution followed by two 3x3 convolutions
        self.branch3 = nn.Sequential(
            nn.Conv2D(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        # 1x1 convolution to adjust output dimensions
        self.conv_adjust = nn.Conv2D(in_channels=192, out_channels=in_channels, kernel_size=1, stride=1, padding=0)

        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=32 * 32 * in_channels, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes)
        )

    def forward(self, x):
        # Main path branches
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)

        # Concatenate the outputs
        concat_out = paddle.concat([branch1_out, branch2_out, branch3_out], axis=1)

        # Adjust output dimensions
        adjusted_out = self.conv_adjust(concat_out)

        # Add the input branch (identity mapping)
        out = paddle.add(x, adjusted_out)

        # Flatten the output for fully connected layers
        out = paddle.flatten(out, start_axis=1, stop_axis=-1)

        # Classification
        out = self.fc_layers(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)