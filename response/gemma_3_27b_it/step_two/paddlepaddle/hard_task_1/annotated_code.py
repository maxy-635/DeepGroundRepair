import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Input image channels
        in_channels = 3  # output tensor shape: (1)
        # Initial convolutional layer
        self.conv1 = nn.Conv2D(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        # Block 1: Channel Attention
        # Path 1: Global Average Pooling
        self.avg_pool = nn.AvgPool2D(kernel_size=7, stride=7)
        self.fc1_1 = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.relu1_1 = nn.ReLU()
        self.fc1_2 = nn.Linear(in_features=in_channels, out_features=in_channels)

        # Path 2: Global Max Pooling
        self.max_pool = nn.MaxPool2D(kernel_size=7, stride=7)
        self.fc2_1 = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.relu2_1 = nn.ReLU()
        self.fc2_2 = nn.Linear(in_features=in_channels, out_features=in_channels)

        # Channel Attention Weight Generation
        self.sigmoid = nn.Sigmoid()

        # Block 2: Spatial Attention
        self.avg_pool2d = nn.AvgPool2D(kernel_size=3, stride=1, padding=1)
        self.max_pool2d = nn.MaxPool2D(kernel_size=3, stride=1, padding=1)
        self.conv1d = nn.Conv1D(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)

        # Additional Branch
        self.conv_add = nn.Conv2D(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.relu_add = nn.ReLU()
        
        # Final Classification
        self.fc_final = nn.Linear(in_features=in_channels * 32 * 32, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Initial Convolution
        x = self.conv1(x)  # output tensor shape: (1, 3, 32, 32)
        x = self.relu1(x)  # output tensor shape: (1, 3, 32, 32)

        # Block 1: Channel Attention
        # Path 1
        path1 = self.avg_pool(x)  # output tensor shape: (1, 3, 4, 4)
        path1 = path1.reshape('mask')
        path1 = self.fc1_1(path1)
        path1 = self.relu1_1(path1)
        path1 = self.fc1_2(path1)

        # Path 2
        path2 = self.max_pool(x)
        path2 = path2.reshape([path2.shape[0], path2.shape[1]])
        path2 = self.fc2_1(path2)
        path2 = self.relu2_1(path2)
        path2 = self.fc2_2(path2)

        # Channel Attention Weights
        channel_attention = self.sigmoid(path1 + path2)
        channel_attention = channel_attention.reshape([channel_attention.shape[0], channel_attention.shape[1], 1, 1])
        x = paddle.multiply(x, channel_attention)

        # Block 2: Spatial Attention
        avg_out = self.avg_pool2d(x)
        max_out = self.max_pool2d(x)
        spatial_features = paddle.concat([avg_out, max_out], axis=1)
        spatial_features = self.conv1d(spatial_features)
        spatial_features = self.sigmoid(spatial_features)
        x = paddle.multiply(x, spatial_features)

        # Additional Branch
        x_add = self.conv_add(x)
        x_add = self.relu_add(x_add)
        x = paddle.add(x, x_add)

        # Final Classification
        x = x.reshape([x.shape[0], -1])
        x = self.fc_final(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)