import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: three average pooling layers with different windows and strides
        self.avg_pool1 = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.avg_pool2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.avg_pool3 = nn.AvgPool2D(kernel_size=4, stride=4, padding=0)

        # Flatten and concatenate the outputs
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=1029, out_features=1024)
        self.reshape = paddle.reshape

        # Second block: four parallel paths
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

        # Path 4: average pooling + 1x1 convolution
        self.path4_pool = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.path4_conv1 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path4_dropout = nn.Dropout(p=0.5)

        # Concatenate paths and final layers
        self.fc2 = nn.Linear(in_features=64*4, out_features=10)  # 4 paths, each with 64 channels

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # First block: three average pooling layers
        pool1 = self.avg_pool1(x)  # output tensor shape: (1, 1, 28, 28)
        pool2 = self.avg_pool2(x)  # output tensor shape: (1, 1, 14, 14)
        pool3 = self.avg_pool3(x)  # output tensor shape: (1, 1, 7, 7)

        # Flatten and concatenate
        pool1_flat = self.flatten(pool1)  # output tensor shape: (1, 784)
        pool2_flat = self.flatten(pool2)  # output tensor shape: (1, 196)
        pool3_flat = self.flatten(pool3)  # output tensor shape: (1, 49)

        # Concatenate along channel dimension (assuming batch dimension is first)
        concat = paddle.concat([pool1_flat, pool2_flat, pool3_flat], axis=1)  # output tensor shape: (1, 1029)

        # Fully connected layer and reshape
        fc_out = self.fc1(concat)
        fc_out = self.reshape(fc_out, shape=[-1, 1, 1, 1024])

        # Second block: four parallel paths
        # Path 1
        path1 = self.path1_conv1(fc_out)
        path1 = self.path1_dropout(path1)

        # Path 2
        path2 = self.path2_conv1(fc_out)
        path2 = self.path2_conv2(path2)
        path2 = self.path2_conv3(path2)
        path2 = self.path2_dropout(path2)

        # Path 3
        path3 = self.path3_conv1(fc_out)
        path3 = self.path3_conv2(path3)
        path3 = self.path3_dropout(path3)

        # Path 4
        path4 = self.path4_pool(fc_out)
        path4 = self.path4_conv1(path4)
        path4 = self.path4_dropout(path4)

        # Concatenate paths along channel dimension
        concat_paths = paddle.concat([path1, path2, path3, path4], axis=1)

        # Final fully connected layers
        fc_out = self.fc2(concat_paths)
        fc_out = paddle.reshape(fc_out, shape=[-1, 10])

        return fc_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])  # Batch size 1, 1 channel, 28x28 MNIST image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)