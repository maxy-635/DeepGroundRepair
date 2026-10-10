import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Channel Group 1: 1x1 Conv2D
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)

        # Channel Group 2: 3x3 Conv2D
        self.conv2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Channel Group 3: 5x5 Conv2D
        self.conv3 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)

        # Concatenation
        self.concat = nn.Concat(dim='mask')

        # Fully Connected Layer 1
        self.fc1 = nn.Linear(in_features=64, out_features=128)
        self.relu1 = nn.ReLU()

        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        # Channel Group 1
        x1 = self.conv1(x)

        # Channel Group 2
        x2 = self.conv2(x1)

        # Channel Group 3
        x3 = self.conv3(x2)

        # Concatenate the three groups
        x = self.concat(x1, x2, x3)

        # Flatten the concatenated tensor
        x = x.flatten(1)

        # Fully Connected Layer 1
        x = self.fc1(x)
        x = self.relu1(x)

        # Fully Connected Layer 2
        x = self.fc2(x)
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor with shape (1, 3, 32, 32) - mimicking the CIFAR-10 input shape
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)