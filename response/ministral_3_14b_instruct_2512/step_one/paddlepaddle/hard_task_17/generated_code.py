import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Global average pooling and weight generation
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))

        # Two fully connected layers with same channel as input
        self.fc1 = nn.Linear(in_features=3*32*32, out_features=3*32*32)
        self.fc2 = nn.Linear(in_features=3*32*32, out_features=3*32*32)

        # Block 2: Feature extraction
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Final classification layers
        self.fc3 = nn.Linear(in_features=128*4*4, out_features=512)
        self.fc4 = nn.Linear(in_features=512, out_features=10)

        # For the weight generation path
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Block 1: Generate weights
        pooled = self.global_avg_pool(x)
        pooled_flat = self.flatten(pooled)
        weights = F.relu6(self.fc2(F.hardswish(self.fc1(pooled_flat))))

        # Reshape weights to match input shape
        weights = weights.reshape((-1, 3, 32, 32))

        # Weighted feature output
        weighted_features = x * weights

        # Block 2: Feature extraction
        features = F.relu6(self.conv1(weighted_features))
        features = self.max_pool(features)
        features = F.relu6(self.conv2(features))

        # Branch connection
        branch = weighted_features

        # Fusion through addition
        fused = features + branch

        # Flatten for classification
        fused_flat = self.flatten(fused)

        # Classification
        out = F.relu6(self.fc3(fused_flat))
        out = self.fc4(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)