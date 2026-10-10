import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1
        self.path1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size='mask', stride='mask', padding='mask')
        self.relu1 = nn.ReLU()

        # Path 2
        self.path2_conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.path2_conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=(1, 7), stride=1, padding=0)
        self.path2_conv3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=(7, 1), stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # Path 3
        self.path3_conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.path3_conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=(1, 7), stride=1, padding=0)
        self.path3_conv3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=(7, 1), stride=1, padding=0)
        self.path3_conv4 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=(1, 7), stride=1, padding=0)
        self.path3_conv5 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=(7, 1), stride=1, padding=0)
        self.relu3 = nn.ReLU()

        # Path 4
        self.path4_pool = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.path4_conv = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.relu4 = nn.ReLU()

        # Fully connected layer
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32 * 32 * 32, out_features=10)  # Assuming input size is 32x32 for CIFAR-10


    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Path 1
        path1 = self.relu1(self.path1(x))  # output tensor shape of this layer should be (1, 32, 26, 26)

        # Path 2
        path2 = self.relu2(self.path2_conv3(self.relu2(self.path2_conv2(self.path2_conv1(x)))))  # output tensor shape: (1, 32, 26, 26)

        # Path 3
        path3 = self.relu3(self.path3_conv5(self.relu3(self.path3_conv4(self.relu3(self.path3_conv3(self.relu3(self.path3_conv2(self.path3_conv1(x)))))))))  # output tensor shape: (1, 32, 20, 20)

        # Path 4
        path4 = self.relu4(self.path4_conv(self.path4_pool(x)))  # output tensor shape: (1, 32, 16, 16)

        # Concatenate paths
        concatenated = paddle.concat([path1, path2, path3, path4], axis=1)

        # Flatten and classify
        flattened = self.flatten(concatenated)
        output = self.linear(flattened)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape) # Expected output shape: [1, 10]