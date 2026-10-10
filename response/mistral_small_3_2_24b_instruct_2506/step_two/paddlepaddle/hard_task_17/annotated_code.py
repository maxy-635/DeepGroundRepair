import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()

        # Block 1: Global average pooling and fully connected layers
        self.block1 = nn.Sequential(
            nn.AdaptiveAvgPool2D(output_size=1),
            nn.Linear(in_features='mask', out_features=64),
            nn.Linear(in_features=64, out_features=64)
        )

        # Block 2: Convolutional layers and max pooling
        self.block2 = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64 * 8 * 8, out_features=128),  # Assuming input features after pooling
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 64, 32, 32)
        # Block 1: Generate weights and apply to input
        weights = self.block1(x)
        weights = paddle.reshape(weights, shape=[-1, 64, 1, 1])
        weighted_features = paddle.multiply(x=x, y=weights)

        # Block 2: Extract deep features
        deep_features = self.block2(x)

        # Fuse the outputs
        fused_features = paddle.add(x=deep_features, y=weighted_features)

        # Classification
        output = self.classifier(fused_features.flatten(start_axis=1))

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 64, 32, 32])  # Example input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]