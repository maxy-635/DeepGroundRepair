import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Increase dimensionality of input channels
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=9, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        # Depthwise separable convolution
        self.depthwise_conv = nn.Conv2D(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1, groups=9)
        self.relu2 = nn.ReLU()

        # Channel Attention
        self.avg_pool = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=9216, out_features=9)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=9, out_features=9)
        self.sigmoid = nn.Sigmoid()

        # 1x1 convolution for dimensionality reduction
        self.conv2 = nn.Conv2D(in_channels=9, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.relu4 = nn.ReLU()

        # Flatten and Fully Connected Layer
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(in_features=3 * 32 * 32, out_features=10)  # Assuming CIFAR-10 image size is 32x32

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Increase dimensionality
        x = self.conv1(x)  # output tensor shape: (1, 9, 32, 32)
        x = self.relu1(x)  # output tensor shape: (1, 9, 32, 32)

        # Depthwise separable convolution
        x = self.depthwise_conv(x)  # output tensor shape: (1, 9, 32, 32)
        x = self.relu2(x)  # output tensor shape: (1, 9, 32, 32)

        # Channel Attention
        attention = self.avg_pool(x)  # output tensor shape: (1, 9, 32, 32)
        attention = self.flatten(attention)  # output tensor shape: (1, 9216)
        attention = self.fc1(attention)
        attention = self.relu3(attention)
        attention = self.fc2(attention)
        attention = self.sigmoid(attention)
        attention = attention.reshape([attention.shape[0], 9, 1, 1])
        x = x * attention

        # Dimensionality reduction
        x = self.conv2(x)
        x = self.relu4(x)

        # Add the initial input
        x = x + self.conv1(x)
        
        # Flatten and Fully Connected Layer
        x = self.flatten(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Dummy input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)