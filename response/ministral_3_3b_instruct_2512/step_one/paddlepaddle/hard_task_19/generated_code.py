import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2)

        # Branch path
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=256)

        # Reshape and multiply
        self.fc_branch_weights = nn.Linear(in_features=256, out_features=256)

        # Final layers
        self.fc3 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        # Main path
        x = self.conv1(x)
        x = paddle.nn.functional.relu(x)
        x = self.conv2(x)
        x = paddle.nn.functional.relu(x)
        x = self.conv3(x)
        x = paddle.nn.functional.relu(x)
        x = self.pool(x)

        # Branch path
        branch = self.global_avg_pool(x)
        branch = paddle.nn.functional.relu(self.fc1(branch))
        branch = paddle.nn.functional.relu(self.fc2(branch))
        branch_weights = paddle.nn.functional.relu(self.fc_branch_weights(branch))
        branch_weights = paddle.reshape(branch_weights, (1, 1, 256))

        # Weight multiplication
        weighted_branch = paddle.multiply(x, branch_weights)

        # Add and concatenate
        x = paddle.add(x, weighted_branch)
        x = paddle.nn.functional.flatten(x)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Example input tensor shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)