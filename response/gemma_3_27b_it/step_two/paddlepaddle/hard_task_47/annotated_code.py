import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block
        self.conv1_1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2D(num_features=32)
        self.conv1_3 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1_3 = nn.BatchNorm2D(num_features=32)
        self.conv1_5 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1_5 = nn.BatchNorm2D(num_features=32)

        # Second block
        self.branch1_conv1 = nn.Conv2D(in_channels=96, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch2_conv1 = nn.Conv2D(in_channels=96, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch2_conv7x7_1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.branch2_conv7x7_2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.branch3_pool = nn.AvgPool2D(kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 10 * 10, out_features=128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First block
        x1_1 = self.bn1_1(self.conv1_1(x))  # output tensor shape: (1, 32, 32, 32)
        x1_3 = self.bn1_3(self.conv1_3(x))  # output tensor shape: (1, 32, 32, 32)
        x1_5 = self.bn1_5(self.conv1_5(x))  # output tensor shape: (1, 32, 32, 32)
        x_block1 = paddle.concat([x1_1, x1_3, x1_5], axis=1)  # output tensor shape: (1, 96, 32, 32)

        # Second block
        x_branch1 = self.branch1_conv1(x_block1)  # output tensor shape: (1, 64, 32, 32)
        x_branch2 = self.branch2_conv1(x_block1)  # output tensor shape: (1, 64, 32, 32)
        x_branch2 = self.branch2_conv7x7_1(x_branch2)  # output tensor shape: (1, 64, 32, 32)
        x_branch2 = self.branch2_conv7x7_2(x_branch2)  # output tensor shape: (1, 64, 32, 32)
        x_branch3 = self.branch3_pool(x_block1)  # output tensor shape: (1, 96, 32, 32)
        x_block2 = paddle.concat([x_branch1, x_branch2, x_branch3], axis=1)  # output tensor shape: (1, 224, 32, 32)
        
        # Fully connected layers
        x = self.flatten(x_block2)  # output tensor shape: (1, 229376)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)