import sys
import paddle
import paddle.nn as nn

class DL_Model(paddle.nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First part: Feature extraction
        self.conv1_1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1_1 = nn.MaxPool1D(kernel_size=3, stride=2, padding=1)

        self.conv1_2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1_2 = nn.MaxPool1D(kernel_size=3, stride=2, padding=1)

        self.conv1_3 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool1_3 = nn.MaxPool1D(kernel_size=3, stride=2, padding=1)

        # Second part: Enhancement
        self.conv2_1 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=0.5)
        self.conv2_2 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        # Upsampling part
        self.up1_1 = nn.Conv2DTranspose(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.skip_conv1_1 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.concat1_1 = nn.Concat(dim=1)

        self.up1_2 = nn.Conv2DTranspose(in_channels=256, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.skip_conv1_2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.concat1_2 = nn.Concat([self.pool1_2(paddle.randn(1,3,32,32)), self.skip_conv1_2(paddle.randn(1,3,32,32))], dim=1)

        self.up1_3 = nn.Conv2DTranspose(in_channels=128, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.skip_conv1_3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.concat1_3 = nn.Concat([self.pool1_1(paddle.randn(1,3,32,32)), self.skip_conv1_3(paddle.randn(1,3,32,32))], dim=1)

        # Final classification
        self.conv_final = nn.Conv2D(in_channels=32, out_channels=10, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(axis=-1)


    def forward(self, x):
        # First part
        x = self.conv1_1(x)
        x = self.pool1_1(x)

        x = self.conv1_2(x)
        x = self.pool1_2(x)

        x = self.conv1_3(x)
        x = self.pool1_3(x)

        # Second part
        x = self.conv2_1(x)
        x = self.dropout(x)
        x = self.conv2_2(x)

        # Upsampling part
        x = self.up1_1(x)
        x = self.skip_conv1_1(x)
        x = self.concat1_1
        x = self.pool1_3(x)

        x = self.up1_2(x)
        x = self.skip_conv1_2(x)
        x = self.concat1_2
        x = self.pool1_2(x)

        x = self.up1_3(x)
        x = self.skip_conv1_3(x)
        x = self.concat1_3
        x = self.pool1_1(x)


        # Final classification
        x = self.conv_final(x)
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = paddle.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (1, 10, 32, 32)