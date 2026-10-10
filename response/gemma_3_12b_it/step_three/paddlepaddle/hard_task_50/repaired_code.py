import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First Block
        self.max_pool1 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.max_pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.max_pool3 = nn.MaxPool2D(kernel_size=4, stride=4, padding=0)
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()
        self.dropout = nn.Dropout2D(p=0.5)
        self.concat = nn.Concat(axis=1)
        self.fc1 = nn.Linear(in_features=1024, out_features=2048)
        self.reshape = nn.Reshape((1, 2048, 1, 1))

        # Second Block
        self.split = nn.Split(num_or_sections=4, axis=-1)
        self.conv1 = nn.Conv2D(in_channels=2048, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2D(in_channels=2048, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2D(in_channels=2048, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2D(in_channels=2048, out_channels=512, kernel_size=7, stride=1, padding=3)
        self.concat2 = nn.Concat()
        self.flatten4 = nn.Flatten()
        self.fc2 = nn.Linear(in_features=2048, out_features=10)

    def forward(self, x):
        # First Block
        x1 = self.max_pool1(x)
        x2 = self.max_pool2(x)
        x3 = self.max_pool3(x)

        x1 = self.flatten1(x1)
        x2 = self.flatten2(x2)
        x3 = self.flatten3(x3)

        x1 = self.dropout(x1)
        x = self.concat((x1, x2, x3))

        x = self.fc1(x)
        x = self.reshape(x)

        # Second Block
        x = self.split(x)
        x1 = F.relu(self.conv1(x[0]))
        x2 = F.relu(self.conv2(x[1]))
        x3 = F.relu(self.conv3(x[2]))
        x4 = F.relu(self.conv4(x[3]))

        x = self.concat2((x1, x2, x3, x4))
        x = self.flatten4(x)

        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.rand(shape=(1, 3, 32, 32))  # Example input: batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print(output_tensor.shape)