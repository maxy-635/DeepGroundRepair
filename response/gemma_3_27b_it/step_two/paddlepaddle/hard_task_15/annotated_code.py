import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distribution import ReshapeTransform

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size='mask')
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features='mask')
        self.reshape = ReshapeTransform(in_event_shape=[64], out_event_shape=[64, 1, 1])

        # Branch Path
        # No modification needed

        # Combined Layers
        self.fc3 = nn.Linear(in_features=64 * 3, out_features=128)  # Input size is 64 (main) + 64 (branch) + 64(original input)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(in_features=128, out_features=10)
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main Path
        x_main = self.conv1(x)  # output tensor shape: (1, 64, 32, 32)
        x_main = self.relu1(x_main)  # output tensor shape: (1, 64, 32, 32)
        x_main = self.conv2(x_main)  # output tensor shape: (1, 64, 32, 32)
        x_main = self.relu2(x_main)  # output tensor shape: (1, 64, 32, 32)
        x_main = self.avg_pool(x_main)  # output tensor shape of this layer should be (1, 64, 32, 32)
        x_main = self.fc1(x_main.flatten(1))
        x_main = self.fc2(x_main)  # output tensor shape of this layer should be (1, 64)
        x_main = self.reshape(x_main)

        # Branch Path
        x_branch = x  # output tensor shape: (1, 3, 32, 32)

        # Combine
        x_combined = paddle.multiply(x_main, x)
        x_combined = paddle.add(x_combined, x_branch)
        x_combined = x_combined.flatten(1)

        x_combined = self.fc3(x_combined)
        x_combined = self.relu3(x_combined)
        x_combined = self.fc4(x_combined)
        x_combined = self.softmax(x_combined)

        return x_combined


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Example input tensor with batch size 1
    output_tensor = model(input_tensor)
    print(output_tensor.shape)