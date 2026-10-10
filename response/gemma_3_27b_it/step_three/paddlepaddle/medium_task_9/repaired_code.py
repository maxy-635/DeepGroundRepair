import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(num_features=16)
        self.relu1 = nn.ReLU()

        # Basic Block 1
        self.conv2 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(num_features=16)
        self.relu2 = nn.ReLU()
        self.conv_branch1 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)

        # Basic Block 2
        self.conv3 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2D(num_features=16)
        self.relu3 = nn.ReLU()
        self.conv_branch2 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)
        
        self.avg_pool = nn.AvgPool2D(kernel_size=8, stride=8, padding=0)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x = self.conv1(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.bn1(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.relu1(x)  # output tensor shape: (1, 16, 32, 32)

        # Basic Block 1
        residual1 = x  # output tensor shape: (1, 16, 32, 32)
        x = self.conv2(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.bn2(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.relu2(x)  # output tensor shape: (1, 16, 32, 32)
        branch1 = self.conv_branch1(residual1)  # output tensor shape: (1, 16, 32, 32)
        x = x + branch1  # output tensor shape: (1, 16, 32, 32)

        # Basic Block 2
        residual2 = x  # output tensor shape: (1, 16, 32, 32)
        x = self.conv3(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.bn3(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.relu3(x)  # output tensor shape: (1, 16, 32, 32)
        branch2 = self.conv_branch2(residual2)  # output tensor shape: (1, 16, 32, 32)
        x = x + branch2  # output tensor shape: (1, 16, 32, 32)
        
        x = self.avg_pool(x)  # output tensor shape: (1, 16, 4, 4)
        x = self.flatten(x)  # output tensor shape: (1, 256)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)