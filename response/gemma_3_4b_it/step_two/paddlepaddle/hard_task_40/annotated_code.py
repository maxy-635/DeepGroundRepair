import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block
        self.pool1 = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.AvgPool2D(kernel_size=4, stride=4, padding=0)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=1 * 1 * 1, out_features=128)
        self.reshape = nn.ReshapeTransform(in_event_shape='mask')

        # Second block
        self.conv1_path1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.dropout1_path1 = nn.Dropout(p=0.25)

        self.conv1_path2 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1_path2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.dropout1_path2 = nn.Dropout(p=0.25)

        self.conv1_path3 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1_path3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.dropout1_path3 = nn.Dropout(p=0.25)

        self.avgpool_path4 = nn.AvgPool2D(kernel_size=2, stride=2, padding=1)
        self.conv1_path4 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.dropout1_path4 = nn.Dropout(p=0.25)

        self.concat = nn.Identity()
        self.fc2 = nn.Linear(in_features=32 * 4, out_features=10)

    def forward(self, x):
        # First block
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.reshape(x)

        # Second block
        path1 = self.conv1_path1(x)
        path1 = self.dropout1_path1(path1)

        path2 = self.conv1_path2(x)
        path2 = self.conv1_path2(path2)
        path2 = self.dropout1_path2(path2)

        path3 = self.conv1_path3(x)
        path3 = self.conv1_path3(path3)
        path3 = self.dropout1_path3(path3)

        path4 = self.avgpool_path4(x)
        path4 = self.conv1_path4(path4)
        path4 = self.dropout1_path4(path4)

        x = self.concat([path1, path2, path3, path4])
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 1, 28, 28))  # Example input shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)