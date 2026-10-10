import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Average pooling layer
        self.avg_pool = F.avg_pool2d
        # 1x1 convolutional layer
        self.conv1 = nn.Conv2D(in_channels=None, # Input channels will be determined dynamically
                                out_channels=32,
                                kernel_size=1,
                                stride=1,
                                padding=0)
        # Flatten layer
        self.flatten = nn.Flatten()
        # Fully connected layer 1
        self.linear1 = nn.Linear(in_features=None, # Input features will be determined dynamically
                                 out_features=128)
        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)
        # Fully connected layer 2
        self.linear2 = nn.Linear(in_features=128,
                                 out_features=10)

    def forward(self, x):
        # Average pooling
        x = self.avg_pool(x, kernel_size=5, stride=3)
        # 1x1 convolution
        x = self.conv1(x)
        # Flatten
        x = self.flatten(x)
        # Fully connected layer 1
        x = self.linear1(x)
        # Dropout
        x = self.dropout(x)
        # Fully connected layer 2
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for MNIST (28x28 grayscale)
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]