import paddle
import paddle.nn as nn
import paddle.vision.transforms as transforms

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial Convolution
        self.conv_initial = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn_initial = nn.BatchNorm2D(num_features=32)
        self.relu_initial = nn.ReLU()

        # Block 1
        self.conv1 = nn.Conv2D(in_channels=32, out_channels='mask', kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(num_features=64)
        self.relu1 = nn.ReLU()

        # Block 2
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(num_features=64)
        self.relu2 = nn.ReLU()

        # Block 3
        self.conv3 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2D(num_features=64)
        self.relu3 = nn.ReLU()

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=128)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Initial Convolution
        x = self.conv_initial(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.bn_initial(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.relu_initial(x)  # output tensor shape: (1, 32, 32, 32)

        # Parallel Blocks
        block1 = self.conv1(x)  # output tensor shape of this layer should be (1, 64, 32, 32)
        block1 = self.bn1(block1)
        block1 = self.relu1(block1)

        block2 = self.conv2(x)  # output tensor shape: (1, 64, 32, 32)
        block2 = self.bn2(block2)  # output tensor shape: (1, 64, 32, 32)
        block2 = self.relu2(block2)  # output tensor shape: (1, 64, 32, 32)

        block3 = self.conv3(x)  # output tensor shape: (1, 64, 32, 32)
        block3 = self.bn3(block3)  # output tensor shape: (1, 64, 32, 32)
        block3 = self.relu3(block3)  # output tensor shape: (1, 64, 32, 32)

        # Add outputs of blocks to initial convolution output
        x = x + block1 + block2 + block3

        # Flatten and Fully Connected Layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)