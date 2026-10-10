import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: 3x3 convolutions
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2)
        )

        # Branch 2: 1x1 convolutions followed by two 3x3 convolutions
        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2)
        )

        # Branch 3: Max pooling
        self.branch3 = nn.Sequential(
            nn.MaxPool2D(kernel_size=2, stride=2)
        )

        # Multi-scale feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2D(output_size=(1, 1))
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process each branch
        branch1_out = self.branch1(x)  # output tensor shape: (1, 64, 16, 16)
        branch2_out = self.branch2(x)  # output tensor shape: (1, 64, 16, 16)
        branch3_out = self.branch3(x)  # output tensor shape: (1, 3, 16, 16)

        # Concatenate outputs from all branches
        fused_features = paddle.concat([branch1_out, branch2_out, branch3_out], axis=1)  # output tensor shape: (1, 131, 16, 16)

        # Apply fusion layers
        fused_features = self.fusion(fused_features)

        # Pass through fully connected layers
        output = self.fc(fused_features)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Example input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)