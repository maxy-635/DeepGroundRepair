import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: parallel branches
        self.conv1x1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=1, padding=1)

        # Concatenation layer
        self.concat = nn.Sequential()

        # Second block: feature processing
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))

        # Weight generation layers
        self.fc1 = nn.Linear(in_features='mask', out_features=64 * 4)
        self.fc2 = nn.Linear(in_features=64*4, out_features=64*4)

        # Final classification layer
        self.fc_output = nn.Linear(in_features=64*4, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (32, 3, 32, 32)
        # First block: parallel branches
        branch1 = self.conv1x1(x)  # output tensor shape: (32, 64, 32, 32)
        branch2 = self.conv3x3(x)  # output tensor shape: (32, 64, 32, 32)
        branch3 = self.conv5x5(x)  # output tensor shape: (32, 64, 32, 32)
        branch4 = self.maxpool(x)  # output tensor shape: (32, 3, 32, 32)

        # Concatenate all branches
        out = paddle.concat([branch1, branch2, branch3, branch4], axis=1)  # output tensor shape: (32, 195, 32, 32)

        # Second block: feature processing
        pooled = self.global_avg_pool(out)  # output tensor shape: (32, 195, 1, 1)
        pooled = paddle.reshape(pooled, [pooled.shape[0], -1])  # output tensor shape: (32, 195)

        # Generate weights
        weights = self.fc1(pooled)
        weights = F.relu(weights)
        weights = self.fc2(weights)
        weights = F.relu(weights)

        # Reshape and multiply
        weights = paddle.reshape(weights, [-1, 256, 1, 1])
        weighted = out * weights

        # Final classification
        weighted = paddle.reshape(weighted, [weighted.shape[0], -1])
        out = self.fc_output(weighted)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([32, 3, 32, 32])  # batch_size=32, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)