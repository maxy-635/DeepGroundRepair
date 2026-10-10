import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path components
        self.global_avg_pool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc1 = nn.Linear(in_features='mask', out_features=512)
        self.fc2 = nn.Linear(512, 512)  # Output channels should match input channels
        self.fc3 = nn.Linear(512, 512)  # For final classification

        # Branch path components
        self.conv_branch = nn.Conv2D(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Final classification layers
        self.fc4 = nn.Linear(512 * 32 * 32, 512)  # Adjust based on actual feature map size
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        pooled = self.global_avg_pool(x)  # output tensor shape: (1, 3, 1, 1)
        pooled = paddle.reshape(pooled, [pooled.shape[0], -1])  # output tensor shape: (1, 3)
        weights = F.relu(self.fc1(pooled))
        weights = F.relu(self.fc2(weights))
        weights = paddle.reshape(weights, [weights.shape[0], x.shape[1], 1, 1])

        # Element-wise multiplication
        weighted = paddle.multiply(x, weights)

        # Branch path
        branch = self.conv_branch(x)

        # Combine paths
        combined = weighted + branch

        # Flatten for final classification
        combined = paddle.reshape(combined, [combined.shape[0], -1])

        # Final classification layers
        out = F.relu(self.fc4(combined))
        out = F.relu(self.fc5(out))
        out = self.fc6(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size 1
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)