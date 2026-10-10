import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First feature extraction block: two convolutional layers followed by max pooling
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)

        # Second feature extraction block: two convolutional layers, one more convolutional layer, followed by max pooling
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)

        # Fully connected layers for classification
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=256 * 4 * 4, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=10)

        # Activation functions and dropout
        self.leaky_relu = nn.LeakyReLU(negative_slope='mask')
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # First feature extraction block
        x = self.leaky_relu(self.conv1(x))  # output tensor shape: (1, 32, 28, 28)
        x = self.leaky_relu(self.conv2(x))  # output tensor shape: (1, 64, 28, 28)
        x = self.pool1(x)  # output tensor shape: (1, 64, 14, 14)

        # Second feature extraction block
        x = self.leaky_relu(self.conv3(x))  # output tensor shape: (1, 128, 14, 14)
        x = self.leaky_relu(self.conv4(x))  # output tensor shape: (1, 128, 14, 14)
        x = self.leaky_relu(self.conv5(x))  # output tensor shape: (1, 256, 14, 14)
        x = self.pool2(x)  # output tensor shape: (1, 256, 7, 7)

        # Flatten and fully connected layers
        x = self.flatten(x)  # output tensor shape: (1, 12544)
        x = self.dropout1(x)  # output tensor shape: (1, 12544)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size=1, channels=1, height=28, width=28)
    input_tensor = paddle.randn((1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print("Input tensor shape:", input_tensor.shape)
    print("Output tensor shape:", output_tensor.shape)