import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.pool1_1 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.pool1_2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.pool1_4 = nn.MaxPool2D(kernel_size=4, stride=4, padding=0)

        self.flatten1_1 = nn.Flatten()
        self.flatten1_2 = nn.Flatten()
        self.flatten1_4 = nn.Flatten()

        self.dropout1_1 = nn.Dropout(p=0.5)
        self.dropout1_2 = nn.Dropout(p=0.5)
        self.dropout1_4 = nn.Dropout(p=0.5)

        # Fully connected layer and reshape
        self.fc1 = nn.Linear(in_features=16*16*1, out_features=64) #Assuming input is 28x28, after pooling 1x1, 14x14, 7x7
        self.reshape = nn.Reshape(shape=[64, 1, 1, 1])

        # Block 2
        self.conv2_1 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2_2_1 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2_2_2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=0)
        self.conv2_2_3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)

        self.conv2_3_1 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2_3_2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=0)
        self.conv2_3_3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)

        self.pool2_4 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.conv2_4 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Final fully connected layers
        self.fc2 = nn.Linear(in_features=32*7*7, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Block 1
        x1_1 = self.pool1_1(x)
        x1_2 = self.pool1_2(x)
        x1_4 = self.pool1_4(x)

        x1_1 = self.flatten1_1(x1_1)
        x1_2 = self.flatten1_2(x1_2)
        x1_4 = self.flatten1_4(x1_4)

        x1_1 = self.dropout1_1(x1_1)
        x1_2 = self.dropout1_2(x1_2)
        x1_4 = self.dropout1_4(x1_4)

        x_block1 = paddle.concat([x1_1, x1_2, x1_4], axis=1)

        # Fully connected layer and reshape
        x_block1 = self.fc1(x_block1)
        x_block1 = self.reshape(x_block1)
        x_block1 = paddle.tile(x_block1, [1, 1, 28, 28])

        # Block 2
        x2_1 = self.conv2_1(x_block1)

        x2_2 = self.conv2_2_1(x_block1)
        x2_2 = self.conv2_2_2(x2_2)
        x2_2 = self.conv2_2_3(x2_2)

        x2_3 = self.conv2_3_1(x_block1)
        x2_3 = self.conv2_3_2(x2_3)
        x2_3 = self.conv2_3_3(x2_3)

        x2_4 = self.pool2_4(x_block1)
        x2_4 = self.conv2_4(x2_4)

        x_block2 = paddle.concat([x2_1, x2_2, x2_3, x2_4], axis=1)

        # Final fully connected layers
        x_block2 = self.flatten1_1(x_block2)
        x_block2 = self.fc2(x_block2)
        x_block2 = F.relu(x_block2)
        x_block2 = self.fc3(x_block2)

        return x_block2


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)