import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Average pooling layers
        self.avg_pool1 = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.avg_pool2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.avg_pool3 = nn.AvgPool2D(kernel_size=4, stride=4, padding=0)

        # Fully connected layer and reshape
        self.fc1 = nn.Linear(in_features=1029, out_features=1024)
        self.reshape = paddle.reshape

        # Second block: Multi-scale feature extraction paths
        # Path 1: 1x1 convolution
        self.path1_conv1 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path1_dropout = nn.Dropout(p=0.5)

        # Path 2: 1x1 + two 3x3 convolutions
        self.path2_conv1 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path2_conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.path2_conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.path2_dropout = nn.Dropout(p=0.5)

        # Path 3: 1x1 + one 3x3 convolution
        self.path3_conv1 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path3_conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.path3_dropout = nn.Dropout(p=0.5)

        # Path 4: Average pooling + 1x1 convolution
        self.path4_pool = nn.AvgPool2D(kernel_size=3, stride=1, padding=1)
        self.path4_conv1 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path4_dropout = nn.Dropout(p=0.5)

        # Final fully connected layers
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # First block: Average pooling and concatenation
        x1 = self.avg_pool1(x)  # output tensor shape: (1, 1, 28, 28)
        x2 = self.avg_pool2(x)  # output tensor shape: (1, 1, 14, 14)
        x3 = self.avg_pool3(x)  # output tensor shape: (1, 1, 7, 7)

        x1_flat = paddle.flatten(x1, start_axis=1)  # output tensor shape: (1, 784)
        x2_flat = paddle.flatten(x2, start_axis=1)  # output tensor shape: (1, 196)
        x3_flat = paddle.flatten(x3, start_axis=1)  # output tensor shape: (1, 49)

        x_concat = paddle.concat([x1_flat, x2_flat, x3_flat], axis=1)  # output tensor shape: (1, 1029)

        # Fully connected layer and reshape
        x_fc = self.fc1(x_concat)
        x_reshape = self.reshape(x_fc, shape=[-1, 1, 32, 32])

        # Second block: Multi-scale feature extraction
        # Path 1
        p1 = self.path1_conv1(x_reshape)
        p1 = self.path1_dropout(p1)

        # Path 2
        p2 = self.path2_conv1(x_reshape)
        p2 = self.path2_conv2(p2)
        p2 = self.path2_conv3(p2)
        p2 = self.path2_dropout(p2)

        # Path 3
        p3 = self.path3_conv1(x_reshape)
        p3 = self.path3_conv2(p3)
        p3 = self.path3_dropout(p3)

        # Path 4
        p4 = self.path4_pool(x_reshape)
        p4 = self.path4_conv1(p4)
        p4 = self.path4_dropout(p4)

        # Concatenate all paths
        x_concat_paths = paddle.concat([p1, p2, p3, p4], axis=1)

        # Final fully connected layers
        x_flat = paddle.flatten(x_concat_paths, start_axis=1)
        x_fc2 = self.fc2(x_flat)
        x_fc3 = self.fc3(x_fc2)

        return x_fc3

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])  # MNIST input shape
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)