import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.pool1_1 = nn.AvgPool2D(kernel_size=1, stride=1)
        self.pool1_2 = nn.AvgPool2D(kernel_size=2, stride=2)
        self.pool1_3 = nn.AvgPool2D(kernel_size=4, stride=4)

        self.dropout1_1 = nn.Dropout(p=0.2)
        self.dropout1_2 = nn.Dropout(p=0.2)
        self.dropout1_3 = nn.Dropout(p=0.2)

        self.flatten1 = nn.Flatten()

        # Concatenate and fuse
        self.concat1 = nn.Concat(dim=-1)
        self.fc1 = nn.Linear(16 * 3, 128)  # Adjust dimension as needed
        self.reshape1 = nn.ReshapeTransform(in_event_shape=(128,))

        # Block 2
        self.conv2_1 = nn.Conv2D(in_channels=16, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = nn.Conv2D(in_channels=16, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2_3 = nn.Conv2D(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.pool2_1 = nn.AvgPool2D(kernel_size=1, stride=1)
        self.conv2_1_dropout = nn.Dropout(p=0.2)
        self.conv2_2_dropout = nn.Dropout(p=0.2)
        self.conv2_3_dropout = nn.Dropout(p=0.2)

        self.concat2 = nn.Concat(dim=-1)

        self.fc2_1 = nn.Linear(128 * 4, 64) # Adjust dimension as needed
        self.fc2_2 = nn.Linear(64, 10)  # 10 classes for MNIST

    def forward(self, x):
        # Block 1
        x = self.pool1_1(x)
        x = self.dropout1_1(x)
        x = self.pool1_2(x)
        x = self.dropout1_2(x)
        x = self.pool1_3(x)
        x = self.dropout1_3(x)

        x = self.flatten1(x)

        # Reshape for Block 2
        x = self.reshape1(x)

        # Block 2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = self.pool2_1(x)

        x = self.concat2(x)

        x = self.fc2_1(x)
        x = self.fc2_2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)