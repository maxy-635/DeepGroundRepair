import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(num_features=16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(num_features=32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        # Concatenation
        self.concat = nn.Concat(dim=1)

        # Convolutional layer after concatenation
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2D(num_features=64)
        self.relu3 = nn.ReLU()

        # Flatten and linear layers
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.concat(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (1, 3, 32, 32)
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)