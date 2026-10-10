import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: Depthwise separable convolution + 1x1 convolution + dropout
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normalized_shape=32),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1),
            nn.Dropout(p=0.2)
        )

        # Branch 2: Depthwise separable convolution + 1x1 convolution + dropout
        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normalized_shape=64),
            nn.Conv2D(in_channels=66, out_channels=64, kernel_size=1, stride=1),
            nn.Dropout(p=0.2)
        )

        # Branch 3: Depthwise separable convolution + 1x1 convolution + dropout
        self.branch3 = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm(normalized_shape=128),
            nn.Conv2D(in_channels=128, out_channels=128, kernel_size=1, stride=1),
            nn.Dropout(p=0.2)
        )

        # Concatenate branches and fully connected layers
        self.concatenate = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32*28*28, out_features=256),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=10)
        )

    def forward(self, x):
        # Process each branch
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)

        # Concatenate outputs
        concatenated = paddle.concat([branch1_out, branch2_out, branch3_out], axis=1)

        # Process concatenated output
        output = self.concatenate(concatenated)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 1, 28, 28))  # Example input shape for MNIST
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)