import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First level: single basic block
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(num_features=16)
        self.relu1 = nn.ReLU()

        # Second level: two residual blocks
        self.conv2_main = nn.Sequential(
            nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=16),
            nn.ReLU()
        )
        self.conv2_branch = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2_add = nn.Sequential(
            nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=16)
        )

        # Third level: global branch
        self.conv3_global = nn.Sequential(
            nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=16),
            nn.ReLU()
        )
        self.conv3_add = nn.Sequential(
            nn.Conv2D(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=16)
        )

        # Final layers
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.fc = nn.Linear(in_features=16, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First level
        x1 = self.conv1(x)  # output tensor shape: (1, 16, 32, 32)
        x1 = self.bn1(x1)  # output tensor shape: (1, 16, 32, 32)
        x1 = self.relu1(x1)  # output tensor shape: (1, 16, 32, 32)

        # Second level
        x2_main = self.conv2_main(x1)  # output tensor shape: (1, 16, 32, 32)
        x2_branch = self.conv2_branch(x1)  # output tensor shape: (1, 16, 32, 32)
        x2 = paddle.add(x2_main, x2_branch)  # output tensor shape: (1, 16, 32, 32)
        x2 = self.conv2_add(x2)

        # Third level
        x3_global = self.conv3_global(x1)
        x3 = paddle.add(x2, x3_global)
        x3 = self.conv3_add(x3)

        # Final layers
        x = self.avg_pool(x3)
        x = paddle.flatten(x, 1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)