import paddle
import paddle.nn as nn

class DL_Model(nn.Sequential):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1),
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1),
            nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=256, out_channels=10, kernel_size=1, stride=1, padding=0),
            nn.Flatten(),
            nn.Linear(in_features=10 * 28 * 28, out_features=200),
            nn.Linear(in_features=200, out_features=10)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.rand(shape=(1, 3, 32, 32))  # Example input tensor (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)