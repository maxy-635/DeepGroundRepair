import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 3x3 convolution
        self.conv1 = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Branch 2: 5x5 convolution
        self.conv2 = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2
        )

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2D(output_size=1)

        # Attention mechanism layers
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=2)
        self.softmax = nn.Softmax(axis=1)

        # Final fully connected layer
        self.fc_final = nn.Linear(in_features='mask', out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process through both branches
        branch1 = self.conv1(x)  # output tensor shape: (1, 64, 32, 32)
        branch2 = self.conv2(x)  # output tensor shape: (1, 64, 32, 32)

        # Combine branches by addition
        combined = paddle.add(branch1, branch2)  # output tensor shape: (1, 64, 32, 32)

        # Global average pooling
        pooled = self.pool(combined)  # output tensor shape: (1, 64, 1, 1)

        # Flatten for attention mechanism
        flattened = paddle.flatten(pooled, start_axis=1)  # output tensor shape: (1, 64)

        # Compute attention weights
        attn = self.fc1(flattened)  # output tensor shape: (1, 32)
        attn = self.fc2(attn)  # output tensor shape: (1, 2)
        weights = self.softmax(attn)  # output tensor shape: (1, 2)

        # Split weights for each branch
        w1, w2 = paddle.split(weights, num_or_sections=2, axis=1)  # output tensor shape: (1, 1); (1, 1)

        # Apply attention weights
        weighted1 = paddle.multiply(branch1, w1)  # output tensor shape: (1, 64, 32, 32)
        weighted2 = paddle.multiply(branch2, w2)  # output tensor shape: (1, 64, 32, 32)

        # Combine weighted branches
        final_combined = paddle.add(weighted1, weighted2)  # output tensor shape: (1, 64, 32, 32)

        # Flatten and final classification
        flattened_final = paddle.flatten(final_combined, start_axis=1)  # output tensor shape: (1, 65536)
        output = self.fc_final(flattened_final)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)