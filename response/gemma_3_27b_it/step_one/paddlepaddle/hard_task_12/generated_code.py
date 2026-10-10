import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path_conv1 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.main_path_conv1x1 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.main_path_conv3x3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

        # Branch Path
        self.branch_path_conv3x3 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Flatten and Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        # Main Path
        x_main = self.main_path_conv1(x)
        x_main1x1 = self.main_path_conv1x1(x_main)
        x_main3x3 = self.main_path_conv3x3(x_main)
        x_main = paddle.concat([x_main1x1, x_main3x3], axis=1)
        x_main = self.relu(x_main)

        # Branch Path
        x_branch = self.branch_path_conv3x3(x)

        # Combine Main and Branch Paths
        x = x_main + x_branch

        # Flatten and Fully Connected Layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 32, 32, 64))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)