import paddle
import paddle.vision.datasets as datasets
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Group 1 Convolution
        self.conv1_1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)

        # Group 2 Convolution
        self.conv2_1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Group 3 Convolution
        self.conv3_1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3_2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Combination
        self.add1 = nn.Add()

        # Fusion with original input
        self.add2 = nn.Add()

        # Flatten and Fully Connected Layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64 * 64, out_features=10)  # Assuming CIFAR-10 has 10 classes

    def forward(self, x):
        # Group 1
        x_group1 = self.conv1_1(x)
        x_group1 = self.conv1_2(x_group1)
        x_group1 = self.conv1_3(x_group1)

        # Group 2
        x_group2 = self.conv2_1(x)
        x_group2 = self.conv2_2(x_group2)
        x_group2 = self.conv2_3(x_group2)

        # Group 3
        x_group3 = self.conv3_1(x)
        x_group3 = self.conv3_2(x_group3)
        x_group3 = self.conv3_3(x_group3)

        # Combine groups
        main_path = self.add1(x_group1, x_group2)
        main_path = self.add2(main_path, x)

        # Flatten and classify
        x = self.flatten(main_path)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)