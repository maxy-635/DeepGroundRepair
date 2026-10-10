import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Split the input into three groups along the last dimension
        self.split = paddle.split

        # Define three depthwise separable convolutional layers with different kernel sizes
        self.conv1x1 = nn.Sequential(
            nn.Conv2D(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask'),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )

        self.conv5x5 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )

        # Flatten the concatenated features
        self.flatten = nn.Flatten()

        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=64 * 32 * 32 * 3, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (32, 3, 32, 32)
        # Split the input into three groups
        x1, x2, x3 = self.split(x, num_or_sections=3, axis=1)  # output tensor shape: (32, 1, 32, 32); (32, 1, 32, 32); (32, 1, 32, 32)

        # Process each group with different convolutional layers
        x1 = self.conv1x1(x1)
        x2 = self.conv3x3(x2)
        x3 = self.conv5x5(x3)

        # Concatenate the outputs
        x = paddle.concat([x1, x2, x3], axis=1)

        # Flatten and pass through the fully connected layer
        x = self.flatten(x)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([32, 3, 32, 32])  # batch_size=32, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [32, 10]