import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Compress input channels with 1x1 convolution
        self.conv1x1_compress = nn.Conv2D(
            in_channels=64,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Two parallel convolutional layers: 1x1 and 3x3
        self.conv1x1_expand = nn.Conv2D(
            in_channels=32,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.conv3x3_expand = nn.Conv2D(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=32 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # Assuming 10 classes for classification

    def forward(self, x):
        # Compress input channels
        x = self.conv1x1_compress(x)

        # Two parallel convolutions
        conv1x1_out = self.conv1x1_expand(x)
        conv3x3_out = self.conv3x3_expand(x)

        # Concatenate along channel dimension
        x = paddle.concat([conv1x1_out, conv3x3_out], axis=1)

        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 64, 32, 32))  # Batch size 1, 64 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)