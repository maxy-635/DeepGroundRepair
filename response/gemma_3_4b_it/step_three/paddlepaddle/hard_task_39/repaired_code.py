import paddle
import paddle.nn as nn
import paddle.vision.datasets as datasets

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.pool1 = nn.MaxPool2D(kernel_size=1, stride=1)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2D(kernel_size=4, stride=4)

        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.LeakyReLU(0.1)

        self.concat1 = nn.Concat(dim=0)

        # Block 2
        self.conv2_1 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=1, stride=1)
        self.relu2_1 = nn.LeakyReLU(0.1)
        self.pixel_shuffle2_1 = nn.functional.pixel_shuffle(self.relu2_1, upscale_factor=2)

        self.conv2_2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2_2 = nn.LeakyReLU(0.1)
        self.pixel_shuffle2_2 = nn.functional.pixel_shuffle(self.relu2_2, upscale_factor=2)

        self.conv2_3 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.relu2_3 = nn.LeakyReLU(0.1)
        self.pixel_shuffle2_3 = nn.functional.pixel_shuffle(self.relu2_3, upscale_factor=2)

        self.pool4 = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 4 * 4, 10)  # Adjust output size based on pixel_shuffle

    def forward(self, x):
        # Block 1
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.relu1(self.conv1(x))
        x = self.concat1([x, x, x])  # Concatenate the three pooling results

        # Block 2
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.pixel_shuffle2_1(x)

        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.pixel_shuffle2_2(x)

        x = self.conv2_3(x)
        x = self.relu2_3(x)
        x = self.pixel_shuffle2_3(x)

        x = self.pool4(x)
        x = self.flatten(x)
        x = self.fc1(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (batch_size=1, channels=1, height=28, width=28)
    input_tensor = paddle.randn(shape=(1, 1, 28, 28), dtype=paddle.float32)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)