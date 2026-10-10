import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class Block(nn.Layer):
    def __init__(self):
        super(Block, self).__init__()
        self.conv = nn.Conv2D(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')






        self.bn = nn.BatchNorm2D(num_features=64)
        self.relu = nn.ReLU()

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        residual = x  # output tensor shape: (1, 65, 28, 28)
        out = self.conv(x)  # output tensor shape: (1, 64, 28, 28)
        out = self.bn(out)  # output tensor shape: (1, 64, 28, 28)
        out = self.relu(out)  # output tensor shape: (1, 64, 28, 28)
        out = paddle.concat([residual, out], axis=1)  # output tensor shape: (1, 65, 28, 28)
        return out

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First pathway
        self.pathway1 = nn.Sequential(
            Block(),
            Block(),
            Block()
        )

        # Second pathway
        self.pathway2 = nn.Sequential(
            nn.Conv2D(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2D(num_features=32),
            nn.ReLU(),
            Block(),
            Block(),
            Block()
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128 * 28 * 28, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Process through both pathways
        out1 = self.pathway1(x)
        out2 = self.pathway2(x)

        # Concatenate outputs from both pathways
        out = paddle.concat([out1, out2], axis=1)

        # Flatten for classification
        out = paddle.flatten(out, start_axis=1)

        # Classification
        out = self.classifier(out)
        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 1, 28, 28))  # Batch of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)