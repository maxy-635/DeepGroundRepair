import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.avg_pool1 = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.avg_pool2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.avg_pool4 = nn.AvgPool2D(kernel_size=4, stride=4, padding=0)

        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(3 * 16, 128)  # Adjust input size based on pooling output

        # Block 2
        self.conv1_1 = nn.Conv2D(128, 32, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2D(128, 32, kernel_size=3, stride=1, padding=0)
        self.conv1_3 = nn.Conv2D(128, 32, kernel_size=1, stride=1, padding=0)
        self.conv1_4 = nn.Conv2D(128, 32, kernel_size=3, stride=1, padding=0)
        self.avg_pool2_1 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.avg_pool2_2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)

        self.concat1 = nn.Concat(dim=0)
        self.fc2 = nn.Linear(3 * 32, 10)

    def forward(self, x):
        # Block 1
        x = self.avg_pool1(x)
        x = self.avg_pool2(x)
        x = self.avg_pool4(x)
        x = self.flatten1(x)
        x = self.fc1(x)
        x = x.reshape(x.shape[0], 128, 1, 1) # Reshape to 4D for Block 2

        # Block 2
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = self.conv1_4(x)
        x = self.avg_pool2_1(x)
        x = self.avg_pool2_2(x)
        x = self.concat1(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = paddle.randn((1, 1, 28, 28))  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)