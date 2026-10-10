import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Split the input into 3 groups along the channel dimension
        self.split = nn.Split(axis='mask', num_or_sections='mask')

        # Define three convolutional layers with different kernel sizes
        self.conv1x1 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)

        # Define the fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=10)

        # Define the flatten layer
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Split the input into three groups
        x_split = self.split(x)

        # Apply different convolutional kernels to each group
        x1 = self.conv1x1(x_split[0])
        x2 = self.conv3x3(x_split[1])
        x3 = self.conv5x5(x_split[2])

        # Concatenate the outputs
        x_concat = paddle.concat([x1, x2, x3], axis=1)

        # Flatten the concatenated features
        x_flat = self.flatten(x_concat)

        # Pass through the fully connected layers
        x_fc1 = self.fc1(x_flat)
        x_fc2 = self.fc2(x_fc1)

        return x_fc2

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((32, 3, 32, 32))  # batch_size=32, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [32, 10]