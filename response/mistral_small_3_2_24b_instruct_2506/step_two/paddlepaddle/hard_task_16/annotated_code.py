import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1x1_1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3_1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1x1_2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        self.conv1x1_3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3_2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1x1_4 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        self.conv1x1_5 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3_3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1x1_6 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Transition Convolution
        self.trans_conv = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Block 2
        self.global_max_pool = nn.MaxPool2D(kernel_size=32, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)

        # Final classification layer
        self.fc_final = nn.Linear(in_features='mask', out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1
        # First group
        x1 = self.conv1x1_1(x)  # output tensor shape: (1, 64, 32, 32)
        x1 = F.relu(x1)  # output tensor shape: (1, 64, 32, 32)
        x1 = self.conv3x3_1(x1)  # output tensor shape: (1, 64, 32, 32)
        x1 = F.relu(x1)  # output tensor shape: (1, 64, 32, 32)
        x1 = self.conv1x1_2(x1)  # output tensor shape: (1, 64, 32, 32)

        # Second group
        x2 = self.conv1x1_3(x)  # output tensor shape: (1, 64, 32, 32)
        x2 = F.relu(x2)  # output tensor shape: (1, 64, 32, 32)
        x2 = self.conv3x3_2(x2)  # output tensor shape: (1, 64, 32, 32)
        x2 = F.relu(x2)  # output tensor shape: (1, 64, 32, 32)
        x2 = self.conv1x1_4(x2)  # output tensor shape: (1, 64, 32, 32)

        # Third group
        x3 = self.conv1x1_5(x)  # output tensor shape: (1, 64, 32, 32)
        x3 = F.relu(x3)  # output tensor shape: (1, 64, 32, 32)
        x3 = self.conv3x3_3(x3)  # output tensor shape: (1, 64, 32, 32)
        x3 = F.relu(x3)  # output tensor shape: (1, 64, 32, 32)
        x3 = self.conv1x1_6(x3)  # output tensor shape: (1, 64, 32, 32)

        # Concatenate the three groups
        x = paddle.concat([x1, x2, x3], axis=1)  # output tensor shape: (1, 192, 32, 32)

        # Transition Convolution
        x = self.trans_conv(x)  # output tensor shape: (1, 64, 32, 32)

        # Block 2
        # Global max pooling
        x_pool = self.global_max_pool(x)  # output tensor shape: (1, 64, 1, 1)
        x_pool = paddle.reshape(x_pool, shape=[x_pool.shape[0], -1])  # output tensor shape: (1, 64)

        # Generate channel-matching weights
        x_weights = self.fc1(x_pool)  # output tensor shape: (1, 64)
        x_weights = F.relu(x_weights)  # output tensor shape: (1, 64)
        x_weights = self.fc2(x_weights)  # output tensor shape: (1, 64)

        # Reshape and multiply
        x_weights = paddle.reshape(x_weights, shape=[x_weights.shape[0], x.shape[1], 1, 1])  # output tensor shape: (1, 64, 1, 1)
        x = x * x_weights  # output tensor shape: (1, 64, 32, 32)

        # Branch connection
        x_branch = x + x  # output tensor shape: (1, 64, 32, 32)

        # Final classification
        x = self.fc_final(x_branch)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)