import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First specialized block
        self.conv_block1 = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2D(kernel_size=2, stride=2, padding=0),
            nn.Dropout(p=0.5)
        )

        # Second specialized block
        self.conv_block2 = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2D(kernel_size=2, stride=2, padding=0),
            nn.Dropout(p=0.5)
        )

        # Global average pooling and classification head
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[32, 1, 28, 28])  # batch_size=32, channels=1, height=28, width=28
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [32, 10]