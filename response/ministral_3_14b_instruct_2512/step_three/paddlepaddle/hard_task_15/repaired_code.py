import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path components
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(num_features=64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(num_features=128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Global average pooling to extract global information
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))

        # Fully connected layers to generate weights
        self.fc1 = nn.Linear(in_features=128, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=3*32*32)  # Matching input dimensions

        # Branch path (direct connection)
        self.branch_conv = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Final classification layers
        self.fc3 = nn.Linear(in_features=128*32*32, out_features=512)
        self.fc4 = nn.Linear(in_features=512, out_features=10)  # CIFAR-10 has 10 classes

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Main path
        x_main = self.relu1(self.bn1(self.conv1(x)))
        x_main = self.pool1(x_main)

        x_main = self.relu2(self.bn2(self.conv2(x_main)))
        x_main = self.pool2(x_main)

        # Global average pooling
        pooled = self.global_avg_pool(x_main)
        pooled = pooled.reshape([pooled.shape[0], -1])  # Flatten

        # Generate weights
        weights = self.fc1(pooled)
        weights = self.fc2(weights)
        weights = weights.reshape([weights.shape[0], 3, 32, 32])  # Reshape to match input

        # Element-wise multiplication with input
        x_main = x_main * weights

        # Branch path (direct connection)
        x_branch = self.branch_conv(x)

        # Combine paths
        x_combined = x_main + x_branch

        # Flatten for classification
        x_combined = x_combined.reshape([x_combined.shape[0], -1])

        # Final classification layers
        x_combined = self.dropout(F.relu(self.fc3(x_combined)))
        x_combined = self.fc4(x_combined)

        return x_combined

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, channels, height, width]
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = paddle.randn(shape=[2, 3, 32, 32])
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)