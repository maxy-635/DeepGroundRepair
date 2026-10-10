import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Path 1: 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Path 2: sequence of convolutions: 1x1, 1x7, 7x1
        self.path2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.Conv2D(in_channels=64, out_channels=66, kernel_size=1, stride=1, padding=0)
        )

        # Align the output dimensions with input image's channel
        self.align = nn.Conv2D(in_channels=130, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Branch directly to input
        self.branch = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=3*32*32, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Path 1 output
        path1_out = self.path1(x)  # output tensor shape: (1, 64, 32, 32)

        # Path 2 output
        path2_out = self.path2(x)  # output tensor shape: (1, 66, 32, 32)

        # Concatenate outputs
        concat_out = paddle.concat([path1_out, path2_out], axis=1)  # output tensor shape: (1, 130, 32, 32)

        # Align dimensions
        aligned_out = self.align(concat_out)

        # Branch output
        branch_out = self.branch(x)

        # Merge outputs
        merged_out = paddle.add(aligned_out, branch_out)

        # Flatten and pass through fully connected layers
        flattened = paddle.flatten(merged_out)
        x = self.fc1(flattened)
        x = paddle.nn.functional.softmax(x, axis=1)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Example input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)