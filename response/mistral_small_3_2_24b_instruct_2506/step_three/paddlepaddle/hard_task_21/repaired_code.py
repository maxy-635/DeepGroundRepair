import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path_conv1 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.main_path_conv2 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.main_path_conv3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.main_path_bn1 = nn.BatchNorm2D(num_features=64)
        self.main_path_bn2 = nn.BatchNorm2D(num_features=64)
        self.main_path_bn3 = nn.BatchNorm2D(num_features=64)
        self.main_path_relu = nn.ReLU()

        # Branch path
        self.branch_conv = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch_bn = nn.BatchNorm2D(num_features=64)

        # Final layers
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split input into three groups along channel dimension
        x1 = x[:, :1, :, :]  # output tensor shape: (1, 1, 32, 32)
        x2 = x[:, 1:2, :, :]  # output tensor shape: (1, 1, 32, 32)
        x3 = x[:, 2:, :, :]  # output tensor shape: (1, 1, 32, 32)

        # Main path
        out1 = self.main_path_conv1(x1)
        out1 = self.main_path_bn1(out1)
        out1 = self.main_path_relu(out1)

        out2 = self.main_path_conv2(x2)
        out2 = self.main_path_bn2(out2)
        out2 = self.main_path_relu(out2)

        out3 = self.main_path_conv3(x3)
        out3 = self.main_path_bn3(out3)
        out3 = self.main_path_relu(out3)

        # Concatenate main path outputs
        main_out = paddle.concat([out1, out2, out3], axis=1)

        # Branch path
        branch_out = self.branch_conv(x)
        branch_out = self.branch_bn(branch_out)

        # Add main and branch outputs
        combined_out = main_out + branch_out

        # Flatten and pass through fully connected layers
        combined_out = paddle.flatten(combined_out, start_axis=1)
        combined_out = self.fc1(combined_out)
        combined_out = F.relu(combined_out)
        combined_out = self.fc2(combined_out)

        return combined_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)