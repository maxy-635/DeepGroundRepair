import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Stage 1: Convolution and Max Pooling
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)

        # Additional convolution and dropout layers
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.25)

        # Stage 2: Upsampling with skip connections
        self.upsample1 = nn.Conv2DTranspose(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.upsample2 = nn.Conv2DTranspose(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Final layers for classification
        self.layer_norm = nn.LayerNorm(normalized_shape=[64])
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=64 * 8 * 8, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Stage 1: Convolution and Max Pooling
        x = self.conv1(x)  # output tensor shape: (1, 32, 32, 32)
        x = nn.ReLU()(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.pool1(x)  # output tensor shape: (1, 32, 16, 16)

        x = self.conv2(x)  # output tensor shape: (1, 64, 16, 16)
        x = nn.ReLU()(x)  # output tensor shape: (1, 64, 16, 16)
        x = self.pool2(x)  # output tensor shape of this layer should be (1, 64, 16, 16)

        # Additional convolution and dropout layers
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.dropout1(x)

        # Stage 2: Upsampling with skip connections
        skip1 = x
        x = self.upsample1(x)  # output tensor shape: (1, 64, 16, 16)
        x = self.conv4(x)  # output tensor shape: (1, 64, 16, 16)
        x = nn.ReLU()(x)  # output tensor shape: (1, 64, 16, 16)
        x = paddle.concat([x, skip1], axis=1)

        skip2 = x
        x = self.upsample2(x)
        x = self.conv5(x)
        x = nn.ReLU()(x)
        x = paddle.concat([x, skip2], axis=1)

        # Final layers for classification
        x = self.layer_norm(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Example input tensor with batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)