import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Feature extraction part 1: 3 pairs of (Conv2D, MaxPool2D)
        self.conv1_1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.conv1_2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.conv1_3 = nn.Conv2D(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')
        self.maxpool3 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Enhancement part: (Conv2D, Dropout, Conv2D)
        self.conv2_1 = nn.Conv2D(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=0.5)
        self.conv2_2 = nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        # Upsampling part: 3 pairs of (Conv2D, Conv2DTranspose) with skip connections
        self.conv3_1 = nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv3_1_t = nn.Conv2DTranspose(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv4_1 = nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_1_t = nn.Conv2DTranspose(in_channels=512, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv5_1 = nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5_1_t = nn.Conv2DTranspose(in_channels=512, out_channels=64, kernel_size=4, stride=2, padding=1)

        # Final classification layer
        self.conv6 = nn.Conv2D(in_channels=64, out_channels=10, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(axis=1)


    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Feature extraction
        x = F.relu(self.conv1_1(x))  # output tensor shape: (1, 64, 32, 32)
        x = self.maxpool1(x)  # output tensor shape: (1, 64, 16, 16)
        x = F.relu(self.conv1_2(x))  # output tensor shape: (1, 128, 16, 16)
        x = self.maxpool2(x)  # output tensor shape: (1, 128, 8, 8)
        x = F.relu(self.conv1_3(x))  # output tensor shape: (1, 256, 8, 8)
        x = self.maxpool3(x)  # output tensor shape: (1, 256, 4, 4)

        # Enhancement
        x = F.relu(self.conv2_1(x))  # output tensor shape: (1, 512, 4, 4)
        x = self.dropout(x)  # output tensor shape: (1, 512, 4, 4)
        x = F.relu(self.conv2_2(x))  # output tensor shape: (1, 512, 4, 4)

        # Upsampling with skip connections
        skip_1 = self.conv1_3(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_1_t(x + skip_1))

        skip_2 = self.conv1_2(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_1_t(x + skip_2))

        skip_3 = self.conv1_1(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_1_t(x + skip_3))

        # Classification
        x = self.conv6(x)
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)