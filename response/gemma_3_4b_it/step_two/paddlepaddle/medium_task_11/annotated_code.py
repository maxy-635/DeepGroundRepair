import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial Convolutional Layer
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)

        # Path 1: Global Average Pooling + FC
        self.pool1 = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.fc1_1 = nn.Linear(64 * 1 * 1, 64)
        self.relu1 = nn.ReLU()
        self.fc1_2 = nn.Linear(64, 10)

        # Path 2: Global Max Pooling + FC
        self.pool2 = nn.AdaptiveMaxPool2D(output_size=(1, 1))
        self.fc2_1 = nn.Linear(64 * 1 * 1, 64)
        self.relu2 = nn.ReLU()
        self.fc2_2 = nn.Linear(64, 10)

        # Channel Attention
        self.add = True
        self.attention = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Initial Convolution
        x = self.conv1(x)

        # Path 1
        x1 = self.pool1(x)
        x1 = x1.reshape(x1.shape[0], -1)
        x1 = self.relu1(self.fc1_1(x1))
        x1 = self.fc1_2(x1)

        # Path 2
        x2 = self.pool2(x)
        x2 = x2.reshape(x2.shape[0], -1)
        x2 = self.relu2(self.fc2_1(x2))
        x2 = self.fc2_2(x2)

        # Add and Attention
        if self.add:
            x = x + x1
        else:
            x = x + x2
        x = self.relu(x)
        x = self.attention(x)

        # Average and Max Pooling
        x_avg = nn.AvgPool2D(kernel_size=8, stride=8, padding=0)(x)
        x_max = nn.MaxPool2D(kernel_size=8, stride=8, padding=0)(x)

        # Concatenate
        x = paddle.concat([x_avg, x_max], axis=1)

        # Element-wise Multiplication
        x = x * x

        # Flatten and Fully Connected
        x = x.flatten(1)
        x = self.fc3(x)

        return x

    def get_submodules(self):
        return [self.conv1, self.pool1, self.fc1_1, self.relu1, self.fc1_2,
                self.pool2, self.fc2_1, self.relu2, self.fc2_2,
                self.attention, self.fc3]

    def set_submodules(self, submodules):
        self.submodules = submodules

    def get_submodule(self, name):
        return self.submodules.get(name)

    def set_submodule(self, name, submodule):
        self.submodules[name] = submodule


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(64, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)