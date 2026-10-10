import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Path 1: Two convolutional blocks followed by average pooling
        self.path1_conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.path1_conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.path1_avg_pool = nn.AvgPool2D(kernel_size=2, stride=2)

        # Path 2: Single convolutional layer
        self.path2_conv = nn.Conv2D(in_channels=3, out_channels=64, kernel_size='mask', stride='mask', padding='mask')

        # Layer normalization for both pathways
        self.layer_norm1 = nn.LayerNorm(normalized_shape=64)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=64)

        # Flatten and fully connected layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64 * 8 * 8, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Path 1
        path1_output = self.path1_conv1(x)  # output tensor shape: (1, 32, 32, 32)
        path1_output = nn.ReLU()(path1_output)  # output tensor shape: (1, 32, 32, 32)
        path1_output = self.path1_conv2(path1_output)  # output tensor shape: (1, 64, 32, 32)
        path1_output = nn.ReLU()(path1_output)  # output tensor shape: (1, 64, 32, 32)
        path1_output = self.path1_avg_pool(path1_output)  # output tensor shape: (1, 64, 16, 16)

        # Path 2
        path2_output = self.path2_conv(x)  # output tensor shape of this layer should be (1, 64, 16, 16)
        path2_output = nn.ReLU()(path2_output)

        # Combine outputs from both pathways
        combined_output = path1_output + path2_output

        # Normalize and flatten
        combined_output = self.layer_norm1(combined_output)
        flattened_output = self.flatten(combined_output)

        # Fully connected layer
        output = self.fc(flattened_output)

        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)