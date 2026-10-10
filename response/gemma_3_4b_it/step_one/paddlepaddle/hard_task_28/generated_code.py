import paddle
import paddle.nn as nn
import paddle.vision.datasets as datasets

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2D(3, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2D(16),
            nn.ReLU(),
            nn.Conv2D(16, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(32),
            nn.ReLU(),
            nn.Conv2D(32, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(32),
            nn.ReLU()
        )

        # Branch path
        self.branch_path = nn.Conv2D(3, 32, kernel_size=3, stride=1, padding=1)

        # Combine paths
        self.combine = nn.Add()

        # Flatten and linear layers
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(32 * 32 * 32, 128)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        main_output = self.main_path(x)
        branch_output = self.branch_path(x)
        combined_output = self.combine(main_output, branch_output)
        flattened_output = self.flatten(combined_output)
        output = self.linear1(flattened_output)
        output = self.linear2(output)
        return output


if __name__ == "__main__":
    # Instantiate the model
    model = DL_Model()

    # Create a dummy input tensor
    input_tensor = paddle.randn(batch_size=1, channels=3, height=32, width=32, dtype=paddle.float32)

    # Pass the input tensor through the model
    output_tensor = model(input_tensor)

    # Print the output shape
    print("Output shape:", output_tensor.shape)