import sys
sys.path.append("./")
from paddle import nn
from paddle.nn import Conv2D, BatchNorm2D, MaxPool2D, Upsample, concat
from paddle import enable_static

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1 = Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn1 = BatchNorm2D(64)
        self.branch1_conv = Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch2_conv = Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch3_conv = Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = MaxPool2D(kernel_size=2, stride=2)
        self.upsample2 = Upsample(size=(None, None, 64), scale_factor=2, mode='nearest')
        self.upsample3 = Upsample(size=(None, None, 64), scale_factor=2, mode='nearest')
        self.conv_concat = Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn_concat = BatchNorm2D(64)

        # Branch path
        self.branch_conv1 = Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn_branch1 = BatchNorm2D(64)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 64 * 64, 1024)
        self.bn_fc1 = BatchNorm2D(1024)
        self.fc2 = nn.Linear(1024, 10)
        self.bn_fc2 = BatchNorm2D(10)

    def forward(self, x):
        # Main path
        main_path = self.conv1(x)
        main_path = self.bn1(main_path)
        main_path = nn.relu(main_path)

        branch1 = self.branch1_conv(x)
        branch1 = self.bn1(branch1)
        branch1 = nn.relu(branch1)

        branch2 = self.branch2_conv(self.maxpool2(x))
        branch2 = self.bn1(branch2)
        branch2 = nn.relu(branch2)
        branch2 = self.upsample2(branch2)

        branch3 = self.branch3_conv(self.maxpool2(x))
        branch3 = self.bn1(branch3)
        branch3 = nn.relu(branch3)
        branch3 = self.upsample3(branch3)

        main_path = concat([main_path, branch1, branch2, branch3], axis=1)
        main_path = self.conv_concat(main_path)
        main_path = self.bn_concat(main_path)
        main_path = nn.relu(main_path)

        # Branch path
        branch = self.branch_conv1(x)
        branch = self.bn_branch1(branch)
        branch = nn.relu(branch)

        # Combine paths
        combined = branch + main_path
        combined = nn.relu(combined)

        # Fully connected layers
        combined = self.fc1(combined)
        combined = self.bn_fc1(combined)
        combined = nn.relu(combined)
        output = self.fc2(combined)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image size
    output_tensor = model(input_tensor)
    print(output_tensor.shape)