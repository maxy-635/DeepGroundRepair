import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Branch 1: 3x3 convolutions
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Branch 2: 1x1 + 3x3 + 3x3 convolutions
        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Branch 3: Max pooling
        self.branch3 = nn.Sequential(
            nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        )

        # Feature fusion
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process through each branch
        branch1_out = self.branch1(x)  # output tensor shape: (1, 64, 32, 32)
        branch2_out = self.branch2(x)  # output tensor shape: (1, 64, 32, 32)
        branch3_out = self.branch3(x)  # output tensor shape of this layer should be (1, 3, 32, 32)

        # Concatenate the outputs
        fused_features = paddle.concat([branch1_out, branch2_out, branch3_out], axis=1)

        # Flatten and pass through fully connected layers
        flattened_features = self.flatten(fused_features)
        fc1_out = self.fc1(flattened_features)
        fc1_out = nn.ReLU()(fc1_out)
        output = self.fc2(fc1_out)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)