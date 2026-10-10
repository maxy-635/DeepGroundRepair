import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distribution import ReshapeTransform

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Average Pooling and Fully Connected Layer
        self.avg_pool1 = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.avg_pool2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.avg_pool3 = nn.AvgPool2D(kernel_size=4, stride=4, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features='mask', out_features=64)
        self.reshape = ReshapeTransform(in_event_shape=[64], out_event_shape=[1, 64, 1, 1])


        # Block 2: Multi-Scale Feature Extraction
        # Path 1: 1x1 Convolution
        self.conv1x1 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.dropout1 = nn.Dropout(p=0.1)

        # Path 2: 1x1 + 3x3 + 3x3 Convolution
        self.conv1x1_path2 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3x3_path2_1 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3x3_path2_2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(p=0.1)

        # Path 3: 1x1 + 3x3 Convolution
        self.conv1x1_path3 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3x3_path3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.dropout3 = nn.Dropout(p=0.1)

        # Path 4: AvgPool + 1x1 Convolution
        self.avg_pool_path4 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.conv1x1_path4 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.dropout4 = nn.Dropout(p=0.1)

        # Fully Connected Layers for Classification
        self.fc2 = nn.Linear(in_features=32 * 4, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Block 1
        x = self.avg_pool1(x)  # output tensor shape: (1, 1, 28, 28)
        x = self.avg_pool2(x)  # output tensor shape: (1, 1, 14, 14)
        x = self.avg_pool3(x)  # output tensor shape: (1, 1, 3, 3)
        x = self.flatten(x)  # output tensor shape: (1, 9)
        x = self.fc1(x)
        x = self.reshape(x)

        # Block 2
        path1 = self.conv1x1(x)
        path1 = self.dropout1(path1)

        path2 = self.conv1x1_path2(x)
        path2 = F.relu(path2)
        path2 = self.conv3x3_path2_1(path2)
        path2 = F.relu(path2)
        path2 = self.conv3x3_path2_2(path2)
        path2 = self.dropout2(path2)

        path3 = self.conv1x1_path3(x)
        path3 = F.relu(path3)
        path3 = self.conv3x3_path3(path3)
        path3 = self.dropout3(path3)

        path4 = self.avg_pool_path4(x)
        path4 = self.conv1x1_path4(path4)
        path4 = self.dropout4(path4)

        x = paddle.concat([path1, path2, path3, path4], axis=1)

        # Classification
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])  # Example input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)