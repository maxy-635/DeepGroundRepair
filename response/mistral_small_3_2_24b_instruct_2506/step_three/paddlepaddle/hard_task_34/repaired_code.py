import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path block
        self.main_path_block = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.ReLU()
        )

        # Repeat the main path block three times
        self.main_path = nn.Sequential(self.main_path_block, self.main_path_block, self.main_path_block)





        # Branch path
        self.branch_path = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Final layers
        self.fuse = nn.Sequential(
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=32*28*28, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Main path
        main_features = self.main_path(x)

        # Branch path
        branch_features = self.branch_path(x)

        # Fuse features
        fused_features = main_features + branch_features
        fused_features = self.fuse(fused_features)

        # Final classification
        flattened_features = self.flatten(fused_features)
        output = self.fc(flattened_features)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)