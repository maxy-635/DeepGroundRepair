import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.AdaptiveAvgPool2D(output_size=1),
            nn.Linear(in_features=256, out_features=128),
            nn.Linear(in_features=128, out_features=256)
        )

        # Final layers
        self.fc1 = nn.Linear(in_features=256 * 4 * 4, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # Main path
        main_out = self.main_path(x)

        # Branch path
        branch_out = self.branch_path(main_out)
        branch_out = paddle.reshape(branch_out, shape=[-1, 256, 1, 1])
        branch_out = paddle.multiply(main_out, branch_out)

        # Combine paths
        combined = paddle.add(main_out, branch_out)
        combined = paddle.flatten(combined, start_axis=1)

        # Final layers
        out = self.fc1(combined)
        out = self.fc2(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]