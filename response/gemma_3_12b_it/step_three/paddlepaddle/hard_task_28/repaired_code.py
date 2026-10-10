import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=3, kernel_size=7, stride=1, padding=3),
            nn.LayerNorm(normalized_shape=[1, 3, 32, 32]),
            nn.Sequential(
                nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0),
                nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
            )
        )
        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        )
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=3 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        main_output = self.main_path(x)
        # Branch path
        branch_output = self.branch_path(x)
        # Combine the outputs
        combined_output = paddle.add(main_output, branch_output)
        # Flatten the output
        flattened_output = nn.Flatten()(combined_output)
        # Fully connected layers
        fc1_output = self.fc1(flattened_output)
        fc2_output = self.fc2(fc1_output)
        return fc2_output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.rand(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)