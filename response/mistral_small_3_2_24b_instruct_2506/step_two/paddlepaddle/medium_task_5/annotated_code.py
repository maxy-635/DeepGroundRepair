import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()

        # Main path
        self.conv1_main = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_main = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1_main = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        self.conv3_main = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2_main = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Branch path
        self.conv1_branch = nn.Conv2D(in_channels=3, out_channels='mask', kernel_size=3, stride=1, padding=1)
        self.pool1_branch = nn.MaxPool2D(kernel_size='mask', stride='mask', padding='mask')

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=128*8*8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        x_main = self.conv1_main(x)  # output tensor shape: (1, 32, 32, 32)
        x_main = self.conv2_main(x_main)  # output tensor shape: (1, 64, 32, 32)
        x_main = self.pool1_main(x_main)  # output tensor shape: (1, 64, 16, 16)
        x_main = self.conv3_main(x_main)  # output tensor shape: (1, 128, 16, 16)
        x_main = self.pool2_main(x_main)  # output tensor shape: (1, 128, 8, 8)

        # Branch path
        x_branch = self.conv1_branch(x)  # output tensor shape of this layer should be (1, 128, 32, 32)
        x_branch = self.pool1_branch(x_branch)  # output tensor shape of this layer should be (1, 16, 32, 32)

        # Combine paths
        x_combined = paddle.add(x_main, x_branch)

        # Fully connected layers
        x_flat = self.flatten(x_combined)
        x_fc1 = self.fc1(x_flat)
        x_fc2 = self.fc2(x_fc1)

        return x_fc2

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)