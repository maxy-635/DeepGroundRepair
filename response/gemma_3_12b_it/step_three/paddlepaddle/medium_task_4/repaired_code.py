import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Path1: Two blocks of convolution followed by average pooling
        self.path1_block1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        )
        self.path1_block2 = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        )

        # Path2: Single convolutional layer
        self.path2 = nn.Conv2D(in_channels=3, out_channels=128, kernel_size=4, stride=4, padding=0)

        # Flatten and fully connected layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=128 * 8 * 8 + 64 * 8 * 8, out_features=10) # Assuming input image size is 32x32

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Path1
        path1_out = self.path1_block1(x)  # output tensor shape: (1, 64, 16, 16)
        path1_out = self.path1_block2(path1_out)  # output tensor shape: (1, 128, 8, 8)

        # Path2
        path2_out = self.path2(x)  # output tensor shape of this layer should be (1, 128, 8, 8)

        # Combine the outputs of both pathways by addition
        combined_out = path1_out + path2_out

        # Flatten the combined output
        flattened_out = self.flatten(combined_out)

        # Fully connected layer for classification
        output = self.fc(flattened_out)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)