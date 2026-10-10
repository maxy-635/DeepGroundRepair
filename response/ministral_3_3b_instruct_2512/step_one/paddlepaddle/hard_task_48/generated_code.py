import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Separable Convolutional Layers with Different Kernel Sizes
        self.conv1_1x1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn1_1x1 = nn.LayerNorm(normalized_shape=64)

        self.conv1_3x3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1_3x3 = nn.LayerNorm(normalized_shape=64)

        self.conv1_5x5 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1_5x5 = nn.LayerNorm(normalized_shape=64)

        # Concatenate outputs of the three groups
        self.concat1 = nn.Sequential(
            nn.Conv2D(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm(normalized_shape=128)
        )

        # Block 2: Four Parallel Branches
        self.path1 = nn.Sequential(
            nn.Conv2D(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        )

        self.path2 = nn.Sequential(
            nn.AvgPool2D(kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        )

        self.path3 = nn.Sequential(
            nn.Conv2D(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.Sequential(
                nn.Conv2D(in_channels=128, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
                nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=1, padding=(1, 0))
            )
        )

        self.path4 = nn.Sequential(
            nn.Conv2D(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.Sequential(
                nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.Sequential(
                    nn.Conv2D(in_channels=128, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
                    nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=1, padding=(1, 0))
                )
            )
        )

        # Concatenate outputs of the four paths
        self.concat2 = nn.Sequential(
            nn.Conv2D(in_channels=496, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.LayerNorm(normalized_shape=256)
        )

        # Final classification layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=256 * 4 * 4, out_features=10)

    def forward(self, x):
        # Block 1: Separable Convolutional Layers
        x1_1x1 = self.bn1_1x1(self.conv1_1x1(x))
        x1_3x3 = self.bn1_3x3(self.conv1_3x3(x))
        x1_5x5 = self.bn1_5x5(self.conv1_5x5(x))

        # Concatenate outputs
        x1 = paddle.concat([x1_1x1, x1_3x3, x1_5x5], axis=1)
        x1 = self.concat1(x1)

        # Block 2: Four Parallel Branches
        path1 = self.path1(x1)
        path2 = self.path2(x1)
        path3 = self.path3(x1)
        path4 = self.path4(x1)

        # Concatenate outputs of the four paths
        x2 = paddle.concat([path1, path2, path3, path4], axis=1)
        x2 = self.concat2(x2)

        # Final classification
        x = self.flatten(x2)
        return self.fc(x)

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)