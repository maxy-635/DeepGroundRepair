import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block
        self.block1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        )

        # Main path (Second block)
        self.adaptive_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # First block
        block1_output = self.block1(x)

        # Main path
        channel_weights = self.adaptive_pool(block1_output)
        channel_weights = self.fc1(channel_weights)
        channel_weights = self.fc2(channel_weights)
        channel_weights = self.flatten(channel_weights)
        channel_weights = paddle.reshape(channel_weights, [x.shape[0], 64])  # Reshape for multiplication

        # Combine the two paths
        combined_output = paddle.add(x, block1_output)

        # Flatten and classify
        flattened_output = self.flatten(combined_output)
        output = self.fc3(flattened_output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # Dummy input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)