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
        self.pool2 = nn.MaxPool2D(kernel_size='mask', stride='mask', padding='mask')

        # Feature processing
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.5)

        # Upsampling with skip connections
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2D(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv5 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Output layer
        self.conv6 = nn.Conv2D(in_channels=32, out_channels=10, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Downsampling stages
        x1 = F.relu(self.conv1(x))  # output tensor shape: (1, 32, 32, 32)
        x1 = self.pool1(x1)  # output tensor shape: (1, 32, 16, 16)

        x2 = F.relu(self.conv2(x1))  # output tensor shape: (1, 64, 16, 16)
        x2 = self.pool2(x2)  # output tensor shape of this layer should be (1, 64, 16, 16)

        # Feature processing
        x3 = F.relu(self.conv3(x2))  # output tensor shape: (1, 128, 8, 8)
        x3 = self.dropout1(x3)  # output tensor shape: (1, 128, 8, 8)

        # Upsampling with skip connections
        x_up1 = self.up1(x3)  # output tensor shape: (1, 128, 16, 16)
        x_up1 = F.relu(self.conv4(x_up1))  # output tensor shape: (1, 64, 16, 16)
        x_up1 = x_up1 + x2

        x_up2 = self.up2(x_up1)
        x_up2 = F.relu(self.conv5(x_up2))
        x_up2 = x_up2 + x1

        # Output layer
        x_out = self.conv6(x_up2)

        return x_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)