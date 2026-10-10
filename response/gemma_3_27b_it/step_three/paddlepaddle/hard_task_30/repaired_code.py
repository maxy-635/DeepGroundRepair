import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First Block: Dual-Path Structure
        self.main_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1)
        )
        self.branch_path = nn.Identity()  # Direct connection

        # Second Block: Depthwise Separable Convolutions
        self.group1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0, groups=3),
            nn.ReLU()
        )
        self.group2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, groups=3),
            nn.ReLU()
        )
        self.group3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2, groups=3),
            nn.ReLU()
        )

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=9216, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First Block
        main_path_output = self.main_path(x)  # output tensor shape: (1, 3, 32, 32)
        branch_path_output = self.branch_path(x)  # output tensor shape: (1, 3, 32, 32)
        combined_output = main_path_output + branch_path_output  # output tensor shape: (1, 3, 32, 32)

        # Second Block
        group1_output = self.group1(combined_output)  # output tensor shape: (1, 3, 32, 32)
        group2_output = self.group2(combined_output)  # output tensor shape: (1, 3, 32, 32)
        group3_output = self.group3(combined_output)  # output tensor shape: (1, 3, 32, 32)

        # Concatenate
        concatenated_output = paddle.concat([group1_output, group2_output, group3_output], axis=1)  # output tensor shape: (1, 9, 32, 32)

        # Flatten
        flattened_output = concatenated_output.reshape((concatenated_output.shape[0], -1))  # output tensor shape: (1, 9216)

        # Fully Connected Layers
        fc1_output = F.relu(self.fc1(flattened_output))
        output = self.fc2(fc1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)