import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: main path with conv layers and branch path
        self.main_path = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

        # Second block: three max pooling layers with different scales
        self.pooling_block = nn.Sequential(
            nn.MaxPool2D(kernel_size=1, stride=1, padding=0),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0),
            nn.MaxPool2D(kernel_size=4, stride=4, padding=0)
        )

        # Flatten and concatenate layers
        self.flatten = nn.Flatten()
        self.concat = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=100),  # First FC layer
            nn.Linear(in_features=100, out_features=10)      # Second FC layer
        )

    def forward(self, x):
        # First block processing
        main_output = self.main_path(x)
        branch_output = x  # Branch path is just the input
        block1_output = main_output + branch_output  # Combine with addition

        # Second block processing
        pool1 = self.pooling_block[0](block1_output)
        pool2 = self.pooling_block[1](block1_output)
        pool3 = self.pooling_block[2](block1_output)

        # Flatten and concatenate
        pool1_flat = self.flatten(pool1)
        pool2_flat = self.flatten(pool2)
        pool3_flat = self.flatten(pool3)

        # Concatenate the flattened vectors
        concatenated = paddle.concat([pool1_flat, pool2_flat, pool3_flat], axis=1)

        # Final classification layers
        output = self.concat(concatenated)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])  # Batch of 1 MNIST image (1 channel, 28x28)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)