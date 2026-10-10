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
        # Process through main path directly without splitting
        main_output = self.main_path(x)

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