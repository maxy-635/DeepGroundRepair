import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.avg_pool = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(in_features='mask', out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)

        # Block 2
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Classification
        self.fc3 = nn.Linear(in_features=64 * 8 * 8, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (4, 3, 32, 32)
        # Block 1
        x1 = self.avg_pool(x)  # output tensor shape: (4, 3, 16, 16)
        x1 = self.fc1(x1)
        x1 = self.fc2(x1)
        x1 = x1.reshape([x.shape[0], 1024, 1, 1])
        x1 = F.multiply(x1, x)

        # Block 2
        x2 = self.conv1(x)
        x2 = F.relu(x2)
        x2 = self.conv2(x2)
        x2 = F.relu(x2)
        x2 = self.max_pool(x2)

        # Fusion
        merged_output = x1 + x2

        # Classification
        x3 = paddle.flatten(merged_output, [1, 2, 3])
        x3 = self.fc3(x3)
        x3 = F.relu(x3)
        output = self.fc4(x3)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.rand(shape=[4, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)