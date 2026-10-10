import paddle
import paddle.vision.datasets as datasets
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Stage 1: Convolution and Max Pooling
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)

        # Stage 2: Convolution and Max Pooling
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)

        # Additional Convolution and Dropout
        self.conv3 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=0.5)

        # Upsampling layers
        self.upsample1 = nn.Upsample(size=(8, 8), scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2D(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.upsample2 = nn.Upsample(size=(32, 32), scale_factor=2, mode='nearest')
        self.conv5 = nn.Conv2D(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Final Convolution and Output
        self.conv_out = nn.Conv2D(in_channels=64, out_channels=10, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Stage 1
        x = self.conv1(x)  # output tensor shape: (1, 64, 32, 32)
        x = self.pool1(x)  # output tensor shape: (1, 64, 16, 16)

        # Stage 2
        x = self.conv2(x)  # output tensor shape: (1, 128, 16, 16)
        x = self.pool2(x)  # output tensor shape: (1, 128, 8, 8)

        # Additional Convolution and Dropout
        x = self.conv3(x)  # output tensor shape: (1, 256, 8, 8)
        x = self.dropout(x)  # output tensor shape: (1, 256, 8, 8)

        # Upsampling
        x = self.upsample1(x)
        x = self.conv4(x)
        x = self.upsample2(x)
        x = self.conv5(x)

        # Final Convolution and Output
        x = self.conv_out(x)
        x = self.sigmoid(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)