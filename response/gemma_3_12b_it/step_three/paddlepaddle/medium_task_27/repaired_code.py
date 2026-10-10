import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: 3x3 kernel
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2D(output_size=(1, 1))
        )

        # Branch 2: 5x5 kernel
        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2D(output_size=(1, 1))
        )

        # Attention Mechanism
        self.attention = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2),
            nn.Softmax(axis=1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process through the two branches
        branch1_output = self.branch1(x)  # output tensor shape: (1, 64, 1, 1)
        branch2_output = self.branch2(x)  # output tensor shape: (1, 64, 1, 1)

        # Attention weights
        attention_weights = self.attention(branch1_output + branch2_output)  # output tensor shape: (1, 2)

        # Weighted combination of the branches
        weighted_output = attention_weights[..., 0] * branch1_output + attention_weights[..., 1] * branch2_output  # output tensor shape: (1, 64, 1, 1)

        # Classification
        output = self.classifier(weighted_output)
        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for CIFAR-10 (3 channels, 32x32)
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)