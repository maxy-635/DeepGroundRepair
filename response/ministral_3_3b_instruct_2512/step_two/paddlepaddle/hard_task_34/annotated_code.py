import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path with three blocks
        self.main_path = nn.Sequential(
            nn.Conv2D(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(32, 32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.ReLU()
        )

        # Three separate blocks for the main path
        self.block1 = nn.Sequential(
            nn.Conv2D(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2DTranspose(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.Conv2D(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2DTranspose(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.block3 = nn.Sequential(
            nn.Conv2D(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2DTranspose(32, 32, kernel_size=3, stride=1, padding=1)
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2D(32, 32, kernel_size=3, stride=1, padding=1)
        )

        # Fusion and classification
        self.fuse = nn.Sequential(
            nn.Add(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        # Process through main path blocks
        x = self.block1(x)
        x = paddle.concat([x, self.block1(x)], axis=1)  # Concatenate along channel dimension

        x = self.block2(x)
        x = paddle.concat([x, self.block2(x)], axis=1)

        x = self.block3(x)
        x = paddle.concat([x, self.block3(x)], axis=1)

        # Branch path
        branch_output = self.branch_path(x)

        # Fusion
        fused_output = self.fuse(x + branch_output)
        return fused_output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 1, 28, 28))  # MNIST input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)