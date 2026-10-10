import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First part: 3 pairs of Conv2D + MaxPool2D
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Second part: Conv2D + Dropout + Conv2D
        self.conv4 = nn.Conv2D(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=0.5)
        self.conv5 = nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        # Third part: 3 pairs of Conv2D + Conv2DTranspose with skip connections
        self.conv6 = nn.Conv2D(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6_t = nn.Conv2DTranspose(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0)

        self.conv7 = nn.Conv2D(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv7_t = nn.Conv2DTranspose(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0)

        self.conv8 = nn.Conv2D(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv8_t = nn.Conv2DTranspose(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)

        # Final 1x1 Conv2D for classification
        self.final_conv = nn.Conv2D(in_channels=64, out_channels=10, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # First part: feature extraction
        x1 = F.relu(self.conv1(x))
        x1 = self.pool1(x1)

        x2 = F.relu(self.conv2(x1))
        x2 = self.pool2(x2)

        x3 = F.relu(self.conv3(x2))
        x3 = self.pool3(x3)

        # Second part: enhance generalization
        x4 = F.relu(self.conv4(x3))
        x4 = self.dropout(x4)
        x4 = F.relu(self.conv5(x4))

        # Third part: upsampling with skip connections
        # Skip connection 1
        x4_up = self.conv6_t(x4)
        x4_up = F.relu(x4_up + x3)

        # Skip connection 2
        x4_up = self.conv7(x4_up)
        x4_up = F.relu(x4_up)
        x4_up = self.conv7_t(x4_up)
        x4_up = F.relu(x4_up + x2)

        # Skip connection 3
        x4_up = self.conv8(x4_up)
        x4_up = F.relu(x4_up)
        x4_up = self.conv8_t(x4_up)
        x4_up = F.relu(x4_up + x1)

        # Final classification
        x = self.final_conv(x4_up)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.reshape((x.shape[0], -1))
        x = F.softmax(x, axis=1)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # CIFAR-10 image shape: (batch_size, 3, 32, 32)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)