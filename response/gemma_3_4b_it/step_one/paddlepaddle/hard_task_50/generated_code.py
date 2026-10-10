import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block
        self.pool1 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.MaxPool2D(kernel_size=4, stride=4, padding=0)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(in_features=1 * 1 * 1 * 4, out_features=128)
        self.reshape = nn.Reshape(in_event_shape=(128, 4))

        # Second block
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=0)

        self.concat = nn.Concat(dim=-1)
        self.flatten2 = nn.Flatten()
        self.fc2 = nn.Linear(in_features=32 * 4, out_features=10)

    def forward(self, x):
        # First block
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.reshape(x)

        # Second block
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.concat(x)
        x = self.flatten2(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)