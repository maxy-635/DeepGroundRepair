import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path = nn.Sequential([
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        ])

        # Branch Path
        self.branch_path = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Fusion
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 1 * 1, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)


    def forward(self, x):
        # Main Path
        main_path_output = self.main_path(x)

        # Branch Path
        branch_path_output = self.branch_path(x)

        # Fusion
        fused_features = main_path_output + branch_path_output

        # Classification
        adaptive_avg_pool_output = self.adaptive_avg_pool(fused_features)
        flatten_output = self.flatten(adaptive_avg_pool_output)
        fc1_output = self.fc1(flatten_output)
        fc1_output = F.relu(fc1_output)
        output = self.fc2(fc1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.rand(shape=(1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)