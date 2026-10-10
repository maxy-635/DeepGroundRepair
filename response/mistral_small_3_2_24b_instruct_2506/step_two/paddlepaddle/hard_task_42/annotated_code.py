import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three parallel paths with different max pooling scales
        self.max_pool1 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.max_pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.max_pool3 = nn.MaxPool2D(kernel_size=4, stride=4, padding=0)

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)

        self.flatten = nn.Flatten()

        # Fully connected layer and reshaping for Block 2
        self.fc1 = nn.Linear(in_features='mask', out_features=128)
        self.reshape = paddle.reshape

        # Block 2: Four parallel paths with different convolution and pooling strategies
        # Path 1: Single 1x1 convolution
        self.conv1_path1 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
        self.conv1_path2 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2_path2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=0)
        self.conv3_path2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=0)

        # Path 3: 1x1 convolution followed by alternating 7x1 and 1x7 convolutions
        self.conv1_path3 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2_path3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=0)
        self.conv3_path3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=0)

        # Path 4: Average pooling followed by 1x1 convolution
        self.avg_pool_path4 = nn.AvgPool2D(kernel_size=3, stride=1, padding=1)
        self.conv1_path4 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Final fully connected layers
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Block 1: Three parallel paths
        x1 = self.max_pool1(x)  # output tensor shape: (1, 1, 28, 28)
        x1 = self.flatten(x1)  # output tensor shape: (1, 784)
        x1 = self.dropout1(x1)  # output tensor shape: (1, 784)

        x2 = self.max_pool2(x)  # output tensor shape: (1, 1, 14, 14)
        x2 = self.flatten(x2)  # output tensor shape: (1, 196)
        x2 = self.dropout2(x2)  # output tensor shape: (1, 196)

        x3 = self.max_pool3(x)  # output tensor shape: (1, 1, 7, 7)
        x3 = self.flatten(x3)  # output tensor shape: (1, 49)
        x3 = self.dropout3(x3)  # output tensor shape: (1, 49)

        # Concatenate the outputs of the three paths
        x = paddle.concat([x1, x2, x3], axis=1)  # output tensor shape: (1, 1029)

        # Fully connected layer and reshaping
        x = self.fc1(x)
        x = self.reshape(x, shape=[-1, 1, 8, 16])

        # Block 2: Four parallel paths
        # Path 1
        path1 = self.conv1_path1(x)

        # Path 2
        path2 = self.conv1_path2(x)
        path2 = self.conv2_path2(path2)
        path2 = self.conv3_path2(path2)

        # Path 3
        path3 = self.conv1_path3(x)
        path3 = self.conv2_path3(path3)
        path3 = self.conv3_path3(path3)

        # Path 4
        path4 = self.avg_pool_path4(x)
        path4 = self.conv1_path4(path4)

        # Concatenate the outputs of the four paths
        x = paddle.concat([path1, path2, path3, path4], axis=1)

        # Flatten and pass through final fully connected layers
        x = self.flatten(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])  # MNIST input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)