import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Conv2D -> ReLU -> MaxPool2D
        self.block1 = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        )
        # Block 2: Conv2D -> ReLU -> MaxPool2D
        self.block2 = nn.Sequential(
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        )

        # AdaptiveAvgPool2D to get a fixed feature vector
        self.adaptive_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))

        # Flatten the output from adaptive pooling
        self.flatten = nn.Flatten()

        # Fully connected layer for classification
        self.linear = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Pass input through the first block
        x1 = self.block1(x)  # output tensor shape: (1, 32, 14, 14)
        # Pass input through the second block
        x2 = self.block2(x)
        # Combine the features from both blocks with the original input
        combined_features = paddle.add(x, x1)
        combined_features = paddle.add(combined_features, x2)

        # Adaptive average pooling
        pooled_features = self.adaptive_pool(combined_features)

        # Flatten the pooled features
        flattened_features = self.flatten(pooled_features)

        # Pass the flattened features through the fully connected layer
        output = self.linear(flattened_features)
        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor with shape (1, 1, 28, 28)
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)