import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Sequential feature extraction layers
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = F.avg_pool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = F.avg_pool2d(kernel_size=2, stride=2, padding=0)

        # Additional convolutional layers
        self.conv3 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2D(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2D(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)

        # Average pooling layer
        self.pool3 = F.avg_pool2d(kernel_size=2, stride=2, padding=0)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=1024 * 7 * 7, out_features=2048)
        self.dropout1 = nn.Dropout2D(p=0.5)
        self.fc2 = nn.Linear(in_features=2048, out_features=1000)
        self.softmax = nn.Softmax2D()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 224, 224])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)