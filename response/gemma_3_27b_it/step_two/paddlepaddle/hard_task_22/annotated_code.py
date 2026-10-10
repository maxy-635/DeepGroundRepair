import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=32),
            nn.ReLU(),
            nn.DepthwiseConv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.BatchNorm2D(num_features=32),
            nn.ReLU(),
            nn.DepthwiseConv2D(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, groups=32),
            nn.BatchNorm2D(num_features=32),
            nn.ReLU()
        )

        # Branch Path
        self.branch_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=32),
            nn.ReLU()
        )

        # Fusion Layer
        self.fusion = nn.Sequential()

        # Classification Layers
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=32 * 32 * 32, out_features=128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        main_output = self.main_path(x)
        branch_output = self.branch_path(x)
        fused_output = main_output + branch_output
        flattened_output = self.flatten(fused_output)
        linear1_output = self.linear1(flattened_output)
        relu_output = self.relu(linear1_output)
        output = self.linear2(relu_output)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)