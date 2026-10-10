import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Parallel branches
        self.conv1x1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.max_pool = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)

        # Second block: Dimensionality reduction and feature weighting
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc_output = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First block
        branch1 = self.conv1x1(x)  # output tensor shape: (1, 64, 32, 32)
        branch2 = self.conv3x3(x)  # output tensor shape: (1, 64, 32, 32)
        branch3 = self.conv5x5(x)  # output tensor shape: (1, 64, 32, 32)
        branch4 = self.max_pool(x)  # output tensor shape of this layer should be (1, 3, 32, 32)

        # Concatenate the branches
        merged_features = paddle.concat([branch1, branch2, branch3, branch4], axis=1)

        # Second block
        pooled_features = self.adaptive_avg_pool(merged_features)
        pooled_features = pooled_features.squeeze()

        fc1_output = self.fc1(pooled_features)
        fc2_output = self.fc2(fc1_output)

        # Reshape the weights to match the input shape
        weights = fc2_output.reshape((1, -1, 1, 1))

        # Element-wise multiplication
        weighted_features = merged_features * weights

        # Final fully connected layer
        output = self.fc_output(weighted_features.squeeze())

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)