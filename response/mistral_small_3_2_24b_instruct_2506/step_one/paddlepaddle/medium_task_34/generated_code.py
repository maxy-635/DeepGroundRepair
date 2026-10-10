import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First part: 3 pairs of <convolutional layer, max-pooling layer>
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Second part: <convolutional layer, Dropout layer, convolutional layer>
        self.conv4 = nn.Conv2D(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=0.5)
        self.conv5 = nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        # Third part: 3 pairs of <convolutional layer, transposed convolutional layer> with skip connections
        self.upconv1 = nn.Conv2D(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.upconv1_transpose = nn.Conv2DTranspose(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0)
        self.upconv2 = nn.Conv2D(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.upconv2_transpose = nn.Conv2DTranspose(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0)
        self.upconv3 = nn.Conv2D(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.upconv3_transpose = nn.Conv2DTranspose(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)

        # Final 1x1 convolutional layer for classification
        self.final_conv = nn.Conv2D(in_channels=64, out_channels=10, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # First part
        x1 = self.conv1(x)
        x1 = self.pool1(x1)
        x2 = self.conv2(x1)
        x2 = self.pool2(x2)
        x3 = self.conv3(x2)
        x3 = self.pool3(x3)

        # Second part
        x4 = self.conv4(x3)
        x4 = self.dropout(x4)
        x4 = self.conv5(x4)

        # Third part with skip connections
        x5 = self.upconv1(x4)
        x5 = self.upconv1_transpose(x5)
        x5 = x5 + x3  # Skip connection

        x6 = self.upconv2(x5)
        x6 = self.upconv2_transpose(x6)
        x6 = x6 + x2  # Skip connection

        x7 = self.upconv3(x6)
        x7 = self.upconv3_transpose(x7)
        x7 = x7 + x1  # Skip connection

        # Final classification
        output = self.final_conv(x7)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # CIFAR-10 input shape: (batch_size, 3, 32, 32)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)