import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Compress channels with a 1x1 convolutional layer
        self.conv1x1 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Parallel convolutional layers
        self.conv1x1_branch = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3x3_branch = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=32 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # Assuming 10 classes

    def forward(self, x):
        # the initial shape of x is: (1, 32, 32, 64)
        # Compress channels
        x = self.conv1x1(x)

        # Parallel branches
        branch1 = self.conv1x1_branch(x)
        branch2 = self.conv3x3_branch(x)

        # Concatenate features
        x = paddle.concat([branch1, branch2], axis=1)

        # Flatten
        x = self.flatten(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 32, 32, 64))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)