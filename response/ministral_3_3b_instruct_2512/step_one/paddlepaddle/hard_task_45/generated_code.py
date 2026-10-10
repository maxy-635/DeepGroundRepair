import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First Block: Split input into three groups and apply depthwise separable convolutions
        self.conv1_1x1 = nn.Conv2D(3, 64, kernel_size=1, stride=1, padding=0)
        self.conv1_3x3 = nn.Conv2D(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_5x5 = nn.Conv2D(3, 64, kernel_size=5, kernel_size=5, stride=1, padding=2)

        # Second Block: Multiple branches for feature extraction
        self.branch1 = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2D(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0),
            nn.Conv2D(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Concatenate outputs from all branches
        self.concat = nn.Sequential(
            nn.Conv2D(192, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 8 * 8, 10)

    def forward(self, x):
        # First Block: Split input into three groups and apply depthwise separable convolutions
        x1 = self.conv1_1x1(x)
        x2 = self.conv1_3x3(x)
        x3 = self.conv1_5x5(x)

        # Concatenate outputs from the first block
        x = paddle.concat([x1, x2, x3], axis=1)

        # Second Block: Multiple branches for feature extraction
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(branch1_out)
        branch3_out = self.branch3(x)

        # Concatenate outputs from all branches
        x = paddle.concat([branch1_out, branch2_out, branch3_out], axis=1)

        # Final layers
        x = self.concat(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)