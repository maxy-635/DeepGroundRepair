import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: Splitting and concatenating with depthwise separable convolutions
        self.conv1x1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn1x1 = nn.BatchNorm(num_channels=64)
        self.conv3x3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3x3 = nn.BatchNorm(num_channels=64)
        self.conv5x5 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn5x5 = nn.BatchNorm(num_channels=64)

        # Second block: Multiple branches for feature extraction
        self.conv1x1_branch1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3_branch1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv1x1_branch2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv1x7_branch2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv7x1_branch2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3_branch2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.avg_pool = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)

        # Fully connected layers for classification
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # First block: Splitting and concatenating with depthwise separable convolutions
        conv1x1_out = self.bn1x1(self.conv1x1(x))
        conv3x3_out = self.bn3x3(self.conv3x3(x))
        conv5x5_out = self.bn5x5(self.conv5x5(x))

        # Concatenate outputs from the three branches
        concatenated = paddle.concat([conv1x1_out, conv3x3_out, conv5x5_out], axis=1)

        # Second block: Multiple branches for feature extraction
        branch1 = self.conv3x3_branch1(self.conv1x1_branch1(concatenated))
        branch2_1x1 = self.conv1x1_branch2(concatenated)
        branch2_1x7 = self.conv1x7_branch2(branch2_1x1)
        branch2_7x1 = self.conv7x1_branch2(branch2_1x7)
        branch2 = self.conv3x3_branch2(branch2_7x1)

        # Average pooling
        pooled = self.avg_pool(branch2)

        # Concatenate all branches
        all_branches = paddle.concat([branch1, branch2_1x1, branch2_1x7, branch2_7x1, branch2, pooled], axis=1)

        # Flatten and pass through fully connected layers
        flattened = self.flatten(all_branches)
        fc1_out = self.fc1(flattened)
        fc2_out = self.fc2(fc1_out)

        return fc2_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))  # Example input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)