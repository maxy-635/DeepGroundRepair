import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 convolution, 3x3 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

        # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

        # Branch 3: Max Pooling
        self.branch3 = nn.Sequential(
            F.max_pool2d(x=None, kernel_size=2, stride=2, padding=0)
        )

        # Dropout layers for each branch
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.25)
        self.dropout3 = nn.Dropout(p=0.25)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # Pass input through each branch
        out1 = self.branch1(x)
        out1 = self.dropout1(out1)

        out2 = self.branch2(x)
        out2 = self.dropout2(out2)

        out3 = self.branch3(x)
        out3 = self.dropout3(out3)

        # Concatenate the outputs
        out = paddle.concat([out1, out2, out3], axis=1)

        # Flatten the output
        out = paddle.flatten(out, start=1)

        # Pass through fully connected layers
        out = self.fc1(out)
        out = paddle.relu(out)
        out = self.fc2(out)
        out = paddle.relu(out)
        out = self.fc3(out)

        return out


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)