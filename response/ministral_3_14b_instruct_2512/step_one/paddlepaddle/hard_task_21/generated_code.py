import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path with three branches using depthwise separable convolutions
        self.main_path = nn.Sequential(
            # First branch (1x1 kernel)
            nn.Sequential(
                nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2D(num_features=3),
                nn.ReLU()
            ),
            # Second branch (3x3 kernel)
            nn.Sequential(
                nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2D(num_features=3),
                nn.ReLU()
            ),
            # Third branch (5x5 kernel)
            nn.Sequential(
                nn.Conv2D(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2D(num_features=3),
                nn.ReLU()
            )
        )

        # Branch path with 1x1 convolution
        self.branch_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=3),
            nn.ReLU()
        )

        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(in_features=3*32*32, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        # Split input into three groups along channel dimension
        x1 = x[:, :1, :, :]  # First channel group
        x2 = x[:, 1:2, :, :]  # Second channel group
        x3 = x[:, 2:, :, :]   # Third channel group

        # Process each group through main path
        x1 = self.main_path[0](x1)
        x2 = self.main_path[1](x2)
        x3 = self.main_path[2](x3)

        # Concatenate along channel dimension
        main_output = paddle.concat([x1, x2, x3], axis=1)

        # Process through branch path
        branch_output = self.branch_path(x)

        # Add main and branch outputs
        combined = main_output + branch_output

        # Flatten for fully connected layers
        flattened = paddle.flatten(combined, start_axis=1)

        # Classification
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, 3, 32, 32] for CIFAR-10
    input_tensor = paddle.randn(shape=[2, 3, 32, 32])
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)