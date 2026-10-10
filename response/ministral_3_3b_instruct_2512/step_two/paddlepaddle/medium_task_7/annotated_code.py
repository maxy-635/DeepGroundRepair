import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First path: three sequential convolutional layers
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Second path: direct convolutional layer
        self.direct_conv = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Layer normalization
        self.norm1 = nn.LayerNorm(normalized_shape='mask')
        self.norm2 = nn.LayerNorm(normalized_shape=64)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64*4*4, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

        # Activation functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First path
        x1 = self.conv1(x)  # output tensor shape: (1, 32, 32, 32)
        x1 = self.relu(x1)  # output tensor shape: (1, 32, 32, 32)
        x1 = self.conv2(x1)  # output tensor shape: (1, 64, 32, 32)
        x1 = self.relu(x1)  # output tensor shape: (1, 64, 32, 32)
        x1 = self.conv3(x1)  # output tensor shape: (1, 64, 32, 32)
        x1 = self.relu(x1)  # output tensor shape: (1, 64, 32, 32)

        # Second path
        x2 = self.direct_conv(x)  # output tensor shape: (1, 64, 32, 32)
        x2 = self.relu(x2)  # output tensor shape: (1, 64, 32, 32)

        # Add outputs
        x = paddle.add(x1, x2)  # output tensor shape: (1, 64, 32, 32)

        # Normalization and dropout
        x = self.norm1(x)
        x = self.dropout(x)

        # Flatten and fully connected layers
        x = x.reshape([-1, 64*4*4])
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    # Create a proper input tensor with shape (batch_size, channels, height, width)
    input_tensor = paddle.rand(shape=[1, 3, 32, 32])  # Using random values for demonstration

    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)