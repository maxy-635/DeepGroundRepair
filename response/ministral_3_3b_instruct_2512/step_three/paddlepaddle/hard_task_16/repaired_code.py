import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1 components
        self.conv1_1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Transition Convolution
        self.trans_conv = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Block 2 components
        self.max_pool = nn.MaxPool2D(kernel_size=8, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=64*4*4, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)

        # Branch
        self.branch_conv = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Final classification layer
        self.final_fc = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1 processing
        x1 = self.conv1_1(x)  # output tensor shape: (1, 64, 32, 32)
        x2 = self.conv1_2(x1)  # output tensor shape: (1, 64, 32, 32)
        x3 = self.conv1_3(x2)  # output tensor shape: (1, 64, 32, 32)
        x = paddle.concat([x1, x2, x3], axis=1)  # output tensor shape: (1, 192, 32, 32)

        # Transition Convolution
        x = self.trans_conv(x)  # output tensor shape: (1, 64, 32, 32)

        # Branch processing
        branch = self.branch_conv(x)

        # Block 2 processing
        x = self.max_pool(x)
        x = paddle.flatten(x, start_dim=1)
        x = paddle.nn.functional.relu(self.fc1(x))
        x = paddle.nn.functional.relu(self.fc2(x))

        # Generate weights for channel-matching
        weights = paddle.reshape(self.fc1(x), [x.shape[0], 64, 1, 1])
        x = paddle.multiply(x, weights)
        x = paddle.sum(x, axis=1)

        # Add branch output
        x = paddle.add(x, branch)

        # Final classification
        x = self.final_fc(x)

        return x

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor with shape (batch_size, channels, height, width)
    # For CIFAR-10, height and width are 32
    input_tensor = paddle.randn((1, 3, 32, 32))

    # Forward pass
    output_tensor = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)