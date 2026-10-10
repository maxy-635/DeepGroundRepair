import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Feature Extraction Path 1: 1x1 convolution
        self.conv1_1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2D(64)
        self.relu1_1 = nn.ReLU()

        # Feature Extraction Path 2: 1x7, 7x1 convolutions
        self.conv2_1 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn2_1 = nn.BatchNorm2D(32)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.bn2_2 = nn.BatchNorm2D(32)
        self.relu2_2 = nn.ReLU()
        self.conv2_3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.bn2_3 = nn.BatchNorm2D(32)
        self.relu2_3 = nn.ReLU()

        # Concatenate the two paths
        self.concat_axis = 1
        self.concat = nn.Concat(dim=self.concat_axis)

        # 1x1 convolution to align dimensions
        self.conv3_1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn3_1 = nn.BatchNorm2D(64)
        self.relu3_1 = nn.ReLU()

        # Branch that connects directly to the input
        self.branch = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn_branch = nn.BatchNorm2D(64)
        self.relu_branch = nn.ReLU()

        # Fully connected layers for classification
        self.fc1 = nn.Linear(64 * 32 * 32, 512)  # Adjust input size based on feature map size
        self.bn_fc1 = nn.BatchNorm2D(512)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)
        self.bn_fc2 = nn.BatchNorm2D(10)


    def forward(self, x):
        # Feature Extraction Path 1
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)

        # Feature Extraction Path 2
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)
        x = self.conv2_3(x)
        x = self.bn2_3(x)
        x = self.relu2_3(x)

        # Concatenate the two paths
        x = self.concat([x, x])

        # 1x1 convolution to align dimensions
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu3_1(x)

        # Branch that connects directly to the input
        input_branch = x
        x = self.branch(input_branch)
        x = self.bn_branch(x)
        x = self.relu_branch(x)

        # Add the branch to the main path
        x = x + input_branch

        # Flatten and fully connected layers
        x = paddle.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)