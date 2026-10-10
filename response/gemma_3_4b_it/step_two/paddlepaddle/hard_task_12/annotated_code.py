import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2D(64, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(32),
            nn.ReLU(),
            nn.Conv2D(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(32),
            nn.ReLU()
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2D(64, 32, kernel_size=3, stride=1, padding=1)
        )

        # Combination
        self.concat = nn.Concat(dim='mask')

        # Flatten and FC layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 32, 10)  # Assuming 10 classes
        self.fc2 = nn.Linear(10, 10)  # Assuming 10 classes

    def forward(self, x):
        main_output = self.main_path(x)
        branch_output = self.branch_path(x)

        combined_output = self.concat([main_output, branch_output])

        flattened_output = self.flatten(combined_output)
        output = self.fc1(flattened_output)
        output = self.fc2(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 32, 32, 64))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)