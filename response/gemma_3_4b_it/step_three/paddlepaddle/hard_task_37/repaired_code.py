import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1_1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1_1 = nn.ReLU()
        self.pool1_1 = nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv1_2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1_2 = nn.ReLU()
        self.pool1_2 = nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv1_3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1_3 = nn.ReLU()

        # Block 2
        self.conv2_1 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2_1 = nn.ReLU()
        self.pool2_1 = nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv2_2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2_2 = nn.ReLU()
        self.pool2_2 = nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv2_3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2_3 = nn.ReLU()

        # Parallel branches
        self.conv_parallel = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu_parallel = nn.ReLU()

        # Combination
        self.add = nn.Add()

        # Flatten and Fully Connected
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64 * 8 * 8, out_features=10)  # Adjust based on image size
        self.celu = nn.CELU(alpha=1.0)

    def forward(self, x):
        # Block 1
        x1 = self.conv1_1(x)
        x1 = self.relu1_1(x1)
        x1 = self.pool1_1(x1)

        x1 = self.conv1_2(x1)
        x1 = self.relu1_2(x1)
        x1 = self.pool1_2(x1)

        x1 = self.conv1_3(x1)
        x1 = self.relu1_3(x1)

        # Block 2
        x2 = self.conv2_1(x1)
        x2 = self.relu2_1(x2)
        x2 = self.pool2_1(x2)

        x2 = self.conv2_2(x2)
        x2 = self.relu2_2(x2)
        x2 = self.pool2_2(x2)

        x2 = self.conv2_3(x2)
        x2 = self.relu2_3(x2)

        # Parallel branch
        x_parallel = self.conv_parallel(x)
        x_parallel = self.relu_parallel(x_parallel)

        # Combination
        x = self.add(x1, x2)
        x = self.add(x, x_parallel)

        # Flatten and Fully Connected
        x = self.flatten(x)
        x = self.fc(x)
        x = self.celu(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = paddle.randn(batch_size=1, channels=1, height=28, width=28)  # Example shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)