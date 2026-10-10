import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0),
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        )

        # Combination and classification
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=128 + 32, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        main_path_output = self.main_path(x)
        branch_path_output = self.branch_path(x)
        combined_output = paddle.add(main_path_output, branch_path_output)
        flattened_output = self.flatten(combined_output)
        fc1_output = self.fc1(flattened_output)
        output = self.fc2(fc1_output)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # Example input tensor (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)