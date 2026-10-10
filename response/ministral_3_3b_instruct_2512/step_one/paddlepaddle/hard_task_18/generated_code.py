import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: two 3x3 convolutional layers followed by average pooling
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AvgPool2D(kernel_size=2, stride=2)

        # Second block: global average pooling to generate channel weights
        self.avg_pool_weight = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.fc1_weight = nn.Linear(in_features=32, out_features=32)
        self.fc2_weight = nn.Linear(in_features=32, out_features=32)

        # Final classification layer
        self.fc_classify = nn.Linear(in_features=32*8*8, out_features=10)

    def forward(self, x):
        # First block processing
        x1 = self.conv1(x)
        x1 = paddle.relu(x1)
        x1 = self.conv2(x1)
        x1 = paddle.relu(x1)
        x1 = self.avg_pool(x1)

        # Add the main path output to the first block output
        x = paddle.add(x, x1)

        # Second block processing
        # Generate channel weights
        weights = self.avg_pool_weight(x)
        weights = paddle.reshape(weights, shape=[-1, 32])
        weights = self.fc1_weight(weights)
        weights = paddle.relu(weights)
        weights = self.fc2_weight(weights)
        weights = paddle.reshape(weights, shape=[-1, 32, 1, 1])

        # Apply weights to the input
        x = paddle.multiply(x, weights)

        # Flatten and classify
        x = paddle.reshape(x, shape=[-1, 32*8*8])
        x = self.fc_classify(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Example input shape for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)