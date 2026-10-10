import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path components
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

        # Branch path component
        self.branch_conv = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Final classification layer
        self.fc = nn.Linear(in_features=3*32*32, out_features=10)  # CIFAR-10 has 10 classes

        # Flatten layer
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Split input into three groups along channel dimension
        x1, x2, x3 = paddle.split(x, num_or_sections=3, axis=1)

        # Main path processing
        # First group remains unchanged
        # Second group goes through conv1
        x2_conv = self.conv1(x2)
        # Combine with third group
        x23 = paddle.concat([x2_conv, x3], axis=1)
        # Additional conv
        x23_conv = self.conv2(x23)
        # Concatenate all three groups
        main_output = paddle.concat([x1, x23_conv], axis=1)

        # Branch path processing
        branch_output = self.branch_conv(x)

        # Combine main and branch outputs
        combined = main_output + branch_output

        # Final classification
        flattened = self.flatten(combined)
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, 3, 32, 32] for CIFAR-10
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)