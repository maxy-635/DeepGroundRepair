import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Compress input channels with 1x1 convolution
        self.conv1x1 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Parallel convolutional layers
        self.conv1x1_parallel = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3x3_parallel = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Flatten and fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=65536, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # Assuming 10 classes for classification

    def forward(self, x):
        # the initial shape of x is: (1, 64, 32, 32)
        # Compress input channels
        x = self.conv1x1(x)  # output tensor shape: (1, 32, 32, 32)

        # Parallel convolutions
        x1 = self.conv1x1_parallel(x)  # output tensor shape: (1, 32, 32, 32)
        x3 = self.conv3x3_parallel(x)  # output tensor shape: (1, 32, 32, 32)

        # Concatenate results
        x = paddle.concat([x1, x3], axis=1)  # output tensor shape: (1, 64, 32, 32)

        # Flatten and pass through fully connected layers
        x = self.flatten(x)  # output tensor shape: (1, 65536)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 64, 32, 32))  # Batch size of 1, 64 channels, 32x32 spatial dimensions
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]