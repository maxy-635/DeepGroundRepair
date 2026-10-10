import paddle
import paddle.nn as nn
import paddle.io
import paddle.vision.datasets as vision_datasets

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolution layer
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Basic block
        self.basic_block = nn.Layer(
            nn.Conv2D(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(16),
            nn.ReLU()
        )

        # Residual connection structure
        self.res1 = self.basic_block
        self.res2 = nn.Layer(
            self.basic_block,
            nn.Conv2D(16, 16, kernel_size=3, stride=1, padding=1)
        )
        self.res3 = nn.Layer(
            self.basic_block,
            nn.Conv2D(16, 16, kernel_size=3, stride=1, padding=1)
        )

        # Global branch
        self.global_conv = nn.Conv2D(16, 16, kernel_size=3, stride=1, padding=1)

        # Average pooling and fully connected layer
        self.avg_pool = nn.AvgPool2D(kernel_size=8, stride=1)
        self.fc = nn.Linear(16 * 8 * 8, 10)

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)

        # First level residual connection
        x1 = self.res1(x)
        x2 = self.res2(x)
        x3 = self.res3(x)
        x = paddle.add(x1, x2 + x3)

        # Second level residual connections
        x4 = self.basic_block(x)
        x5 = self.basic_block(x)
        x = paddle.add(x4, x5)

        # Third level global branch
        x6 = self.global_conv(x)
        x = paddle.add(x6, x)

        # Average pooling and fully connected layer
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = paddle.randn(batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)