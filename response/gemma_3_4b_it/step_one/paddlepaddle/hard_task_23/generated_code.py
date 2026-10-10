import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2D(64)

        # Branch 1: Local feature extraction
        self.conv2_1 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2D(32)
        self.conv2_2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2D(32)

        # Branch 2: Downsampling and upsampling
        self.avgpool2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.conv3_1 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2D(64)
        self.conv3_2 = nn.Conv2DTranspose(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2D(64)

        # Branch 3: Downsampling and upsampling
        self.avgpool3 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.conv4_1 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2D(64)
        self.conv4_2 = nn.Conv2DTranspose(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2D(64)

        # Concatenation
        self.concat_axis = 1
        self.conv5 = nn.Conv2D(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2D(64)

        # Fully connected layer
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        # Initial convolutional layer
        x = self.bn1(self.conv1(x))
        x = nn.relu(x)

        # Branch 1
        x1 = self.bn2_1(self.conv2_1(x))
        x1 = nn.relu(x1)
        x1 = self.bn2_2(self.conv2_2(x1))
        x1 = nn.relu(x1)

        # Branch 2
        x2 = self.avgpool2(x)
        x2 = self.bn3_1(self.conv3_1(x2))
        x2 = nn.relu(x2)
        x2 = self.bn3_2(self.conv3_2(x2))
        x2 = nn.relu(x2)

        # Branch 3
        x3 = self.avgpool3(x)
        x3 = self.bn4_1(self.conv4_1(x3))
        x3 = nn.relu(x3)
        x3 = self.bn4_2(self.conv4_2(x3))
        x3 = nn.relu(x3)

        # Concatenation
        x = paddle.concat([x1, x2, x3], axis=self.concat_axis)

        # Refinement
        x = self.bn5(self.conv5(x))
        x = nn.relu(x)

        # Fully connected layer
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    dummy_input = paddle.randn(1, 3, 32, 32)
    output_tensor = model(dummy_input)
    print(output_tensor.shape)