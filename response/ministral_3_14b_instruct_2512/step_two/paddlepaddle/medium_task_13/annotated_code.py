import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(num_features=32)
        self.relu1 = nn.ReLU()

        # Second convolutional layer
        self.conv2 = nn.Conv2D(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')
        self.bn2 = nn.BatchNorm2D(num_features=64)
        self.relu2 = nn.ReLU()

        # Third convolutional layer
        self.conv3 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2D(num_features=128)
        self.relu3 = nn.ReLU()

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=512)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First convolutional block
        x1 = self.conv1(x)  # output tensor shape: (1, 32, 32, 32)
        x1 = self.bn1(x1)  # output tensor shape: (1, 32, 32, 32)
        x1 = self.relu1(x1)  # output tensor shape: (1, 32, 32, 32)

        # Second convolutional block with concatenation
        x2 = paddle.concat([x1, x], axis=1)  # output tensor shape: (1, 35, 32, 32)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)

        # Third convolutional block with concatenation
        x3 = paddle.concat([x2, x1], axis=1)
        x3 = self.conv3(x3)
        x3 = self.bn3(x3)
        x3 = self.relu3(x3)

        # Flatten and fully connected layers
        x = self.flatten(x3)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)