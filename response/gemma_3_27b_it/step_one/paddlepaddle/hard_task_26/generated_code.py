import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path_conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

        # Branch 1
        self.branch1_conv = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Branch 2
        self.branch2_pool = nn.MaxPool2D(kernel_size=2, stride=2)
        self.branch2_conv = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch2_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Branch 3
        self.branch3_pool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.branch3_conv = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.main_path_conv2 = nn.Conv2D(in_channels=96, out_channels=16, kernel_size=1, stride=1, padding=0)

        # Branch Path
        self.branch_path_conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=16 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Main Path
        x = self.main_path_conv1(x)
        x = self.relu(x)

        branch1 = self.branch1_conv(x)

        branch2 = self.branch2_pool(x)
        branch2 = self.branch2_conv(branch2)
        branch2 = self.branch2_upsample(branch2)

        branch3 = self.branch3_pool(x)
        branch3 = self.branch3_conv(branch3)
        branch3 = self.branch3_upsample(branch3)

        x = paddle.concat([branch1, branch2, branch3], axis=1)
        x = self.main_path_conv2(x)
        x = self.relu(x)

        # Branch Path
        branch_path = self.branch_path_conv1(x)

        # Combine Paths
        x = x + branch_path

        # Flatten and Fully Connected Layers
        x = x.reshape([x.shape[0], -1])
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)