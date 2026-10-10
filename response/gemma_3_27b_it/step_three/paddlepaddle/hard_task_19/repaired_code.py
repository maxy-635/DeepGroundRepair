import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from paddle.distribution import ReshapeTransform
import paddle
import paddle.add as add

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        )

        # Branch Path
        self.branch_path = nn.Sequential(
            nn.AvgPool2D(kernel_size=8, stride=8, padding=0),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32)
        )

        # Final Classification Layers
        self.reshape_transform = ReshapeTransform(in_event_shape=[32], out_event_shape=[1, 32])
        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=512, out_features=10)


    def forward(self, x):
        # Main Path
        main_output = self.main_path(x)

        # Branch Path
        branch_output = self.branch_path(x)
        branch_output = self.reshape_transform(branch_output)

        # Multiply branch output with input
        branch_output_expanded = paddle.tile(branch_output, [1, 1, 8, 8])
        multiplied_output = x * branch_output_expanded

        # Add outputs from both paths
        combined_output = add.add(main_output, multiplied_output)

        # Flatten and pass through final layers
        flattened_output = paddle.reshape(combined_output, [combined_output.shape[0], -1])
        fc1_output = self.fc1(flattened_output)
        relu1_output = self.relu1(fc1_output)
        output = self.fc2(relu1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.to_tensor(np.random.rand(1, 3, 32, 32), dtype='float32')
    output_tensor = model(input_tensor)
    print(output_tensor.shape)