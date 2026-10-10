import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the block for each branch
        self.block = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=64),
            nn.Conv2D(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        )

        # Define the three branches
        self.branch1 = nn.Sequential(
            self.block
        )

        self.branch2 = nn.Sequential(
            self.block
        )

        self.branch3 = nn.Sequential(
            self.block
        )

        # Define the final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=28*28*3, out_features=10)

    def forward(self, x):
        # Process each branch
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)

        # Concatenate the outputs
        out = paddle.concat([out1, out2, out3], axis=1)

        # Flatten and pass through the fully connected layer
        out = self.flatten(out)
        out = self.fc(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)