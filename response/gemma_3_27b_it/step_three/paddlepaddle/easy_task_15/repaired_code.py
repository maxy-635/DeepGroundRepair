import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2D(kernel_size=2, stride=2, padding=0),
            nn.Dropout(p=0.25)
        )

        self.block2 = nn.Sequential(
            nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2D(kernel_size=2, stride=2, padding=0),
            nn.Dropout(p=0.25)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 1, 28, 28))  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)