import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()
        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2D(kernel_size='mask', stride='mask', padding='mask')
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        )

        # Combined layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 8 * 8 + 32 * 8 * 8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process through main path
        main_path_output = self.main_path(x)  # output tensor shape of this layer should be (1, 64, 32, 32)

        # Process through branch path
        branch_path_output = self.branch_path(x)  # output tensor shape: (1, 32, 32, 32)

        # Concatenate the outputs of the two paths
        combined_output = paddle.concat([main_path_output, branch_path_output], axis=1)

        # Flatten the combined output
        flattened_output = self.flatten(combined_output)

        # Pass through fully connected layers
        fc1_output = self.fc1(flattened_output)
        output = self.fc2(fc1_output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor with shape (1, 3, 32, 32)
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)