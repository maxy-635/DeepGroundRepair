import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        )
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        )
        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2D(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask'),
            nn.ReLU()
        )
        # Separate convolutional layer
        self.conv_direct = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # Flatten layer
        self.flatten = nn.Flatten()
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=256 * 8 * 8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Path 1: First two convolutional blocks
        out1 = self.conv1(x)  # output tensor shape: (1, 64, 16, 16)
        out2 = self.conv2(out1)  # output tensor shape: (1, 128, 8, 8)
        # Path 2: Third convolutional block
        out3 = self.conv3(x)
        # Path 3: Direct path
        out_direct = self.conv_direct(x)
        # Add outputs from all paths
        added_output = paddle.add(out2, paddle.add(out3, out_direct))
        # Flatten the output
        flattened_output = self.flatten(added_output)
        # Fully connected layers
        fc1_output = self.fc1(flattened_output)
        output = self.fc2(fc1_output)
        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor with shape [1, 3, 32, 32]
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)