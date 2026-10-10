import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Feature extraction layers
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.avgpool1 = nn.AvgPool2D(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.avgpool2 = nn.AvgPool2D(kernel_size=2, stride=2)

        # Additional convolutional layers
        self.conv3 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2D(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2D(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.avgpool3 = nn.AvgPool2D(kernel_size=2, stride=2)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=512 * 28 * 28, out_features=4096)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.dropout2 = nn.Dropout(p=0.5)

        # Output layer
        self.output = nn.Linear(in_features=4096, out_features=1000)

    def forward(self, x):
        # Feature extraction layers
        x = self.conv1(x)
        x = self.avgpool1(x)

        x = self.conv2(x)
        x = self.avgpool2(x)

        # Additional convolutional layers
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool3(x)

        # Fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)

        # Output layer
        x = self.output(x)
        x = F.softmax(x, axis=1)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 224, 224))  # batch_size=1, channels=3, height=224, width=224
    output_tensor = model(input_tensor)
    print(output_tensor.shape)