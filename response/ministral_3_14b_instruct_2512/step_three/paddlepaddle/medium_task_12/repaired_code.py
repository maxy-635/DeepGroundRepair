import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block
        self.conv1_1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2D(num_features=64)
        self.relu1_1 = nn.ReLU()

        # Second block
        self.conv2_1 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2D(num_features=128)
        self.relu2_1 = nn.ReLU()

        # Third block
        self.conv3_1 = nn.Conv2D(in_channels=192, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2D(num_features=256)
        self.relu3_1 = nn.ReLU()

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=114688, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First block
        x1 = self.relu1_1(self.bn1_1(self.conv1_1(x)))  # output tensor shape: (1, 64, 32, 32)

        # Second block (concatenate with previous output)
        x2 = self.relu2_1(self.bn2_1(self.conv2_1(x1)))  # output tensor shape: (1, 128, 32, 32)
        x2 = paddle.concat([x1, x2], axis=1)  # output tensor shape: (1, 192, 32, 32)

        # Third block (concatenate with previous output)
        x3 = self.relu3_1(self.bn3_1(self.conv3_1(x2)))  # output tensor shape: (1, 256, 32, 32)
        x3 = paddle.concat([x2, x3], axis=1)  # output tensor shape: (1, 448, 32, 32)

        # Pooling to reduce spatial dimensions
        x3 = F.max_pool2d(x3, kernel_size=2, stride=2)  # output tensor shape: (1, 448, 16, 16)

        # Flatten and fully connected layers
        x = self.flatten(x3)  # output tensor shape: (1, 114688)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, channels, height, width]
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)