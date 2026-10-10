import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the convolutional layers with different kernel sizes
        self.conv1x1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)

        # Define the max pooling layer
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=1, padding=1)

        # Define the flatten layer
        self.flatten = nn.Flatten()

        # Define the fully connected layers
        self.fc1 = nn.Linear(in_features='mask', out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Apply the convolutional layers and max pooling
        x1 = self.conv1x1(x)  # output tensor shape: (1, 64, 32, 32)
        x2 = self.conv3x3(x)  # output tensor shape: (1, 64, 32, 32)
        x3 = self.conv5x5(x)  # output tensor shape: (1, 64, 32, 32)
        x4 = self.maxpool(x)  # output tensor shape: (1, 3, 32, 32)

        # Concatenate the features
        x = paddle.concat([x1, x2, x3, x4], axis=1)  # output tensor shape: (1, 195, 32, 32)

        # Flatten and apply the fully connected layers
        x = self.flatten(x)  # output tensor shape: (1, 199680)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # CIFAR-10 image shape: (3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]