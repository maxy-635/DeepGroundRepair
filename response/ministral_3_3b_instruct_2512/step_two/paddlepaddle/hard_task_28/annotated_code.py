import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.depthwise_conv = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.layer_norm = nn.LayerNorm(normalized_shape=32)
        self.pointwise_conv1 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.pointwise_conv2 = nn.Conv2D(in_channels=32, out_channels=10, kernel_size=1, stride=1, padding=0)

        # Branch path
        self.branch_conv = nn.Conv2D(in_channels=3, out_channels='mask', kernel_size=1, stride=1, padding=0)

        # Flatten and fully connected layers
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=32*8*8, out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        x_main = self.depthwise_conv(x)  # output tensor shape: (1, 32, 32, 32)
        x_main = self.layer_norm(x_main)  # output tensor shape: (1, 32, 32, 32)
        x_main = self.pointwise_conv1(x_main)  # output tensor shape: (1, 32, 32, 32)
        x_main = self.pointwise_conv2(x_main)  # output tensor shape: (1, 10, 32, 32)

        # Branch path
        x_branch = self.branch_conv(x)  # output tensor shape of this layer should be (1, 10, 32, 32)

        # Combine paths
        x_combined = paddle.add(x_main, x_branch)

        # Flatten and fully connected layers
        x_flattened = self.flatten(x_combined)
        x_flattened = paddle.nn.functional.relu(x_flattened)
        x_flattened = self.linear1(x_flattened)
        x_flattened = paddle.nn.functional.relu(x_flattened)
        x_flattened = self.linear2(x_flattened)

        return x_flattened

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Example input shape for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)