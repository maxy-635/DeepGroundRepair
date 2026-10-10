import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Downsampling stages
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Additional processing
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.5)

        # Upsampling stages with skip connections
        self.upconv1 = nn.Conv2D(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upconv2 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

        # Final output layer
        self.output_conv = nn.Conv2D(in_channels=32, out_channels=10, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Downsampling
        x1 = F.relu(self.conv1(x))
        x1 = self.pool1(x1)
        x2 = F.relu(self.conv2(x1))
        x2 = self.pool2(x2)

        # Additional processing
        x3 = F.relu(self.conv3(x2))
        x3 = self.dropout1(x3)

        # Upsampling with skip connections
        x4 = F.relu(self.upconv1(x3))
        x4 = self.upsample1(x4)
        x4 = paddle.concat([x4, x2], axis=1)  # Skip connection
        x5 = F.relu(self.upconv2(x4))
        x5 = self.upsample2(x5)
        x5 = paddle.concat([x5, x1], axis=1)  # Skip connection

        # Final output
        x6 = self.output_conv(x5)
        x6 = F.softmax(x6, axis=1)

        return x6

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)