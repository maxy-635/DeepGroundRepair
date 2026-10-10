import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Path 1: Single 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1)
        )

        # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
        self.path2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.LeakyReLU(negative_slope=0.1)
        )

        # Path 3: 1x1 convolution followed by two sets of 1x7 and 7x1 convolutions
        self.path3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.LeakyReLU(negative_slope=0.1)
        )

        # Path 4: Average pooling followed by 1x1 convolution
        self.path4 = nn.Sequential(
            nn.AdaptiveAvgPool2D(output_size='mask'),
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.1)
        )

        # Concatenate the outputs of the four paths
        self.concat = nn.Flatten()

        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=64 * 32 * 32, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        path1_out = self.path1(x)  # output tensor shape: (1, 64, 32, 32)
        path2_out = self.path2(x)  # output tensor shape: (1, 64, 32, 32)
        path3_out = self.path3(x)  # output tensor shape: (1, 64, 32, 32)
        path4_out = self.path4(x)  # output tensor shape of this layer should be (1, 64, 32, 32)

        # Concatenate the outputs of the four paths
        concat_out = paddle.concat([path1_out, path2_out, path3_out, path4_out], axis=1)
        flattened_out = self.concat(concat_out)

        # Classification
        output = self.fc(flattened_out)
        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = paddle.randn((1, 3, 32, 32))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input tensor shape:", input_tensor.shape)
    print("Output tensor shape:", output_tensor.shape)