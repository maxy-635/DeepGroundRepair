import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Three groups with different kernel sizes
        self.group1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        self.group2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        self.group3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        # Second block: Multiple branches for feature extraction
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2D(kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=196608, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (32, 3, 32, 32)
        # First block: Concatenate outputs from three groups
        x1 = self.group1(x)  # output tensor shape: (32, 64, 32, 32)
        x2 = self.group2(x)  # output tensor shape: (32, 64, 32, 32)
        x3 = self.group3(x)  # output tensor shape: (32, 64, 32, 32)
        x = paddle.concat([x1, x2, x3], axis=1)  # output tensor shape: (32, 192, 32, 32)

        # Second block: Concatenate outputs from three branches
        x1 = self.branch1(x)  # output tensor shape: (32, 64, 32, 32)
        x2 = self.branch2(x)  # output tensor shape: (32, 64, 32, 32)
        x3 = self.branch3(x)  # output tensor shape: (32, 64, 32, 32)
        x = paddle.concat([x1, x2, x3], axis=1)  # output tensor shape: (32, 192, 32, 32)

        # Flatten and pass through fully connected layers
        x = paddle.flatten(x, start_axis=1)  # output tensor shape: (32, 196608)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((32, 3, 32, 32))  # batch_size=32, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (32, 10)