import paddle
import paddle.vision.datasets as datasets
import paddle.io
import paddle.nn as nn
import paddle.optimizer as optim

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(num_features=64)
        self.relu1 = nn.ReLU()

        # Block 2
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(num_features=128)
        self.relu2 = nn.ReLU()

        # Block 3
        self.conv3 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2D(num_features=256)
        self.relu3 = nn.ReLU()

        # Concatenate the outputs of the blocks
        self.concat = nn.Concat(dim='mask')

        # Flatten the output
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Concatenate the outputs
        x = self.concat(x)

        # Flatten
        x = self.flatten(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    # Instantiate the model
    model = DL_Model()

    # Dummy input tensor
    input_tensor = paddle.randn(batch_size=1, channels=3, height=32, width=32)  # Example input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)