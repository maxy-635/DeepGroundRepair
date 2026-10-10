import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 convolution for dimensionality reduction
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

        # Branch 3: 1x1 convolution followed by 5x5 convolution
        self.branch3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        )

        # Branch 4: 3x3 max pooling followed by 1x1 convolution
        self.branch4 = nn.Sequential(
            nn.MaxPool2D(kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Pass input through each branch
        branch1_out = self.branch1(x)  # output tensor shape: (1, 64, 32, 32)
        branch2_out = self.branch2(x)  # output tensor shape: (1, 64, 32, 32)
        branch3_out = self.branch3(x)  # output tensor shape: (1, 64, 32, 32)
        branch4_out = self.branch4(x)  # output tensor shape of this layer should be (1, 64, 32, 32)

        # Concatenate the outputs of all branches
        concatenated = paddle.concat([branch1_out, branch2_out, branch3_out, branch4_out], axis=1)

        # Flatten the features
        flattened = nn.Flatten()(concatenated)

        # Pass through fully connected layers
        fc1_out = self.fc1(flattened)
        output = self.fc2(fc1_out)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.rand(shape=[1, 3, 32, 32])  # Dummy input tensor (batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)