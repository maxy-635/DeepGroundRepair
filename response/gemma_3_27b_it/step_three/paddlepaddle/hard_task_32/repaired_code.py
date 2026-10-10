import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the specialized block
        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2D(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias_attr=True),
                nn.ReLU(),
                nn.Dropout(p=0.25),
                nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias_attr=True),
                nn.ReLU(),
                nn.Dropout(p=0.25)
            )

        # Create three branches
        self.branch1 = block(1, 16)
        self.branch2 = block(1, 32)
        self.branch3 = block(1, 64)

        # Concatenate the outputs
        self.concat = nn.Sequential(
            nn.Conv2D(in_channels=16 + 32 + 64, out_channels=128, kernel_size=3, stride=1, padding=1, bias_attr=True),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=25088, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Pass input through the branches
        out1 = self.branch1(x)  # output tensor shape: (1, 16, 28, 28)
        out2 = self.branch2(x)  # output tensor shape: (1, 32, 28, 28)
        out3 = self.branch3(x)  # output tensor shape: (1, 64, 28, 28)

        # Concatenate the outputs
        out = paddle.concat([out1, out2, out3], axis=1)  # output tensor shape: (1, 112, 28, 28)

        # Process through the concatenated layer
        out = self.concat(out)  # output tensor shape: (1, 128, 14, 14)
        out = paddle.reshape(out, [out.shape[0], -1])  # output tensor shape: (1, 25088)

        # Fully connected layers
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)