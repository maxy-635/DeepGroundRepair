import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1 parameters
        self.conv1_1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2D(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Transition convolution
        self.trans_conv = nn.Conv2D(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')

        # Block 2 parameters
        self.global_pool = nn.MaxPool2D(kernel_size=3, stride=3, padding=0)
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)

        # Final classification layer
        self.fc_out = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1
        # Split into three groups along channel dimension
        x1 = self.conv1_1(x)  # output tensor shape: (1, 64, 32, 32)
        x2 = self.conv1_2(x1)  # output tensor shape: (1, 64, 32, 32)
        x3 = self.conv1_3(x2)  # output tensor shape: (1, 3, 32, 32)

        # Concatenate along channel dimension
        block1_out = paddle.concat([x1, x2, x3], axis=1)  # output tensor shape: (1, 131, 32, 32)

        # Transition convolution
        trans_out = self.trans_conv(block1_out)

        # Block 2 - main path
        # Global max pooling
        pooled = self.global_pool(trans_out)

        # Flatten for FC layers
        pooled_flat = paddle.flatten(pooled, start_axis=1, stop_axis=-1)

        # Generate channel-matching weights
        weights = F.relu6(self.fc1(pooled_flat))
        weights = F.hardswish(self.fc2(weights))

        # Reshape weights to match trans_out shape
        weights = paddle.reshape(weights, [-1, 64, 1, 1])

        # Multiply with trans_out
        main_path = trans_out * weights

        # Block 2 - branch path (direct connection)
        branch = trans_out

        # Add main path and branch
        combined = main_path + branch

        # Final classification
        out = self.fc_out(paddle.flatten(combined, start_axis=1, stop_axis=-1))

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)