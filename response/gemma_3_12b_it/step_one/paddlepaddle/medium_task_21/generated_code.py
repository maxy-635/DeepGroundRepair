import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 convolution
        self.branch1 = nn.Conv2D(
            in_channels=3,  # Assuming CIFAR-10 input has 3 channels (RGB)
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.dropout1 = nn.Dropout(p=0.25)

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2D(
                in_channels=3,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.Conv2D(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
        )
        self.dropout2 = nn.Dropout(p=0.25)

        # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
        self.branch3 = nn.Sequential(
            nn.Conv2D(
                in_channels=3,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.Conv2D(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.Conv2D(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
        )
        self.dropout3 = nn.Dropout(p=0.25)

        # Branch 4: Average pooling followed by 1x1 convolution
        self.branch4 = nn.Sequential(
            F.avg_pool2d(x=None, kernel_size=3, stride=2, padding=1),  # Placeholder, will be defined in forward
            nn.Conv2D(
                in_channels=3,  # Assuming CIFAR-10 input has 3 channels (RGB)
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0
            ),
        )
        self.dropout4 = nn.Dropout(p=0.25)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)  # Adjusted based on expected output size
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # Branch 1
        x1 = self.branch1(x)
        x1 = self.dropout1(x1)

        # Branch 2
        x2 = self.branch2(x)
        x2 = self.dropout2(x2)

        # Branch 3
        x3 = self.branch3(x)
        x3 = self.dropout3(x3)

        # Branch 4
        x4 = self.branch4[0](x)
        x4 = self.branch4[1](x4)
        x4 = self.dropout4(x4)

        # Concatenate the outputs of all branches
        x = paddle.concat([x1, x2, x3, x4], axis=1)

        # Flatten the tensor
        x = nn.Flatten()(x)

        # Fully connected layers
        x = self.fc1(x)
        x = paddle.relu(x)
        x = self.fc2(x)
        x = paddle.relu(x)
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]