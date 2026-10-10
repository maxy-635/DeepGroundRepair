import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.initial_conv = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Branch 1
        self.branch1_conv = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 2
        self.branch2_pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.branch2_conv = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch2_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Branch 3
        self.branch3_pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.branch3_conv = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Concatenation and final convolution
        self.concat_conv = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Branch path
        self.branch_path_conv = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # Main path
        x_main = self.initial_conv(x)

        # Branch 1
        branch1 = self.branch1_conv(x_main)

        # Branch 2
        branch2 = self.branch2_pool(x_main)
        branch2 = self.branch2_conv(branch2)
        branch2 = self.branch2_upsample(branch2)

        # Branch 3
        branch3 = self.branch3_pool(x_main)
        branch3 = self.branch3_conv(branch3)
        branch3 = self.branch3_upsample(branch3)

        # Concatenate branches
        x_main = paddle.concat([branch1, branch2, branch3], axis=1)
        x_main = self.concat_conv(x_main)

        # Branch path
        x_branch = self.branch_path_conv(x)

        # Add main path and branch path
        x = x_main + x_branch

        # Flatten and fully connected layers
        x = paddle.flatten(x, start_axis=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # CIFAR-10 image shape: (3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (1, 10)