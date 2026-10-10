import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Sequential blocks
        self.block1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=128),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=256),
            nn.ReLU()
        )

        # Parallel branch
        self.parallel_conv = nn.Conv2D(in_channels=3, out_channels=256, kernel_size=1, stride=1, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features='mask', out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process through sequential blocks
        out1 = self.block1(x)  # output tensor shape: (1, 64, 32, 32)
        out2 = self.block2(out1)  # output tensor shape: (1, 128, 32, 32)
        out3 = self.block3(out2)  # output tensor shape: (1, 256, 32, 32)

        # Process through parallel branch
        parallel_out = self.parallel_conv(x)  # output tensor shape: (1, 256, 32, 32)

        # Add all outputs
        combined = paddle.add(out3, parallel_out)  # output tensor shape: (1, 256, 32, 32)

        # Flatten and pass through fully connected layers
        combined = paddle.flatten(combined, start_axis=1)  # output tensor shape: (1, 262144)
        combined = self.fc1(combined)
        combined = self.fc2(combined)

        return combined

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)