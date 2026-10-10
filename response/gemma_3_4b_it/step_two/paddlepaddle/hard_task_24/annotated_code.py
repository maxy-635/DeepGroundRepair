import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 3x3 Conv2D
        self.branch1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu1 = nn.ReLU()

        # Branch 2: MaxPool2D, 3x3 Conv2D, Upsample
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(32)
        self.relu2 = nn.ReLU()
        self.upsample2 = nn.Upsample(size=(32, 32), scale_factor=2, mode='nearest')

        # Branch 3: MaxPool2D, 3x3 Conv2D, Upsample
        self.pool3 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2D(32)
        self.relu3 = nn.ReLU()
        self.upsample3 = nn.Upsample(size=(32, 32), scale_factor=2, mode='nearest')

        # Concatenate branches
        self.concat = nn.Concat(dim='mask')

        # 1x1 Conv2D
        self.conv_final = nn.Conv2D(in_channels=128, out_channels=10, kernel_size=1, stride=1, padding=0)

        # Fully Connected layers
        self.fc1 = nn.Linear(10 * 32 * 32, 512)
        self.bn_fc1 = nn.BatchNorm2D(512)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm2D(256)
        self.relu_fc2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # Branch 1
        x1 = self.branch1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        # Branch 2
        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)
        x2 = self.upsample2(x2)

        # Branch 3
        x3 = self.pool3(x)
        x3 = self.conv3(x3)
        x3 = self.bn3(x3)
        x3 = self.relu3(x3)
        x3 = self.upsample3(x3)

        # Concatenate
        x = self.concat([x1, x2, x3])

        # 1x1 Conv2D
        x = self.conv_final(x)

        # Fully Connected layers
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu_fc2(x)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)