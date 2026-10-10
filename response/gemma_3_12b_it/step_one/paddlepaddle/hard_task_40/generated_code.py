import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block
        self.avgpool1 = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.avgpool2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.avgpool3 = nn.AvgPool2D(kernel_size=4, stride=4, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=1, out_features=1024)
        self.reshape = nn.Reshape((1024, 1, 1, 1))  # Reshape to 4D tensor

        # Second block - Parallel paths
        self.conv2d_path1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2d_path2 = nn.Sequential(
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
        self.conv2d_path3 = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
        self.conv2d_path4 = nn.Sequential(
            nn.AvgPool2D(kernel_size=2, stride=2, padding=0),
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        )
        self.dropout = nn.Dropout(p=0.5)

        # Output layers
        self.fc2 = nn.Linear(in_features=64 * 4, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)


    def forward(self, x):
        # First block
        x = self.avgpool1(x)
        x = self.avgpool2(x)
        x = self.avgpool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.reshape(x)

        # Second block - Parallel paths
        x_path1 = self.conv2d_path1(x)
        x_path1 = self.dropout(x_path1)

        x_path2 = self.conv2d_path2(x)
        x_path2 = self.dropout(x_path2)

        x_path3 = self.conv2d_path3(x)
        x_path3 = self.dropout(x_path3)

        x_path4 = self.conv2d_path4(x)
        x_path4 = self.dropout(x_path4)

        # Concatenate paths
        x = paddle.concat([x_path1, x_path2, x_path3, x_path4], axis=1)

        # Output layers
        x = self.fc2(x)
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.rand(shape=(1, 1, 28, 28))  # Dummy input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)