import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First specialized block
        self.block1 = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=1, stride=1),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            nn.AvgPool2D(kernel_size=2, stride=2),
            nn.Dropout(p=0.2)
        )

        # Second specialized block
        self.block2 = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            nn.AvgPool2D(kernel_size=2, stride=2),
            nn.Dropout(p=0.2)
        )

        # Global pooling and classification
        self.global_avg_pool = nn.AvgPool2D(kernel_size=7, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # MNIST images are 28x28 with 1 channel
    input_tensor = paddle.randn((1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)