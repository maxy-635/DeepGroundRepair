import paddle
import paddle.nn as nn
import paddle.vision.datasets as datasets
import paddle.vision.transforms as transforms
import numpy as np

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2D(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(16),
            nn.ReLU(),
            nn.Conv2D(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(32),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels='mask', kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(32),
            nn.ReLU()
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2D(3, 8, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(8),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32 * 32 * 32, 10)  # Adjust output size based on CIFAR-10

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        main_output = self.main_path(x)  # output tensor shape of this layer should be (1, 8, 32, 32)
        branch_output = self.branch_path(x)  # output tensor shape: (1, 8, 32, 32)

        combined_output = paddle.add(main_output, branch_output)

        flattened_output = self.flatten(combined_output)
        output = self.linear(flattened_output)

        return output


if __name__ == "__main__":
    # Instantiate the model
    model = DL_Model()

    # Create a dummy input tensor
    input_tensor = paddle.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    
    # Pass the input through the model
    output_tensor = model(input_tensor)

    # Print the output shape
    print("Output shape:", output_tensor.shape)