import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Step 1: Compress input channels with a 1x1 convolutional layer
        self.conv1x1 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Step 2: Expand features through two parallel convolutional layers
        self.conv1x1_parallel = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3x3_parallel = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Step 3: Concatenate the results of the two parallel convolutional layers
        self.concat = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Step 4: Flatten the output feature map and pass through fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # Assuming 10 classes for classification

    def forward(self, x):
        # the initial shape of x is: (1, 32, 32, 64)
        # Step 1: Compress input channels
        x = self.conv1x1(x)

        # Step 2: Expand features through parallel convolutional layers
        conv1x1_out = self.conv1x1_parallel(x)
        conv3x3_out = self.conv3x3_parallel(x)

        # Step 3: Concatenate the outputs
        x = paddle.concat([conv1x1_out, conv3x3_out], axis=1)
        x = self.concat(x)

        # Step 4: Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = paddle.relu(self.fc1(x))
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 32, 32, 64))  # Dummy input tensor with shape (1, 32, 32, 64)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)