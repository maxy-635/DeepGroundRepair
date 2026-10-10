import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1x1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.dropout = nn.Dropout(p=0.5)

        # Block 2
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2D(kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=262144, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1
        x1 = self.conv1x1(x)  # output tensor shape: (1, 64, 32, 32)
        x2 = self.conv3x3(x)  # output tensor shape: (1, 64, 32, 32)
        x3 = self.conv5x5(x)  # output tensor shape: (1, 64, 32, 32)
        x = paddle.concat([x1, x2, x3], axis=1)  # output tensor shape: (1, 192, 32, 32)
        x = self.dropout(x)  # output tensor shape: (1, 192, 32, 32)

        # Block 2
        b1 = self.branch1(x)  # output tensor shape: (1, 64, 32, 32)
        b2 = self.branch2(x)  # output tensor shape: (1, 64, 32, 32)
        b3 = self.branch3(x)  # output tensor shape: (1, 64, 32, 32)
        b4 = self.branch4(x)  # output tensor shape: (1, 64, 32, 32)
        x = paddle.concat([b1, b2, b3, b4], axis=1)  # output tensor shape: (1, 256, 32, 32)

        # Final layers
        x = self.flatten(x)  # output tensor shape: (1, 262144)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]