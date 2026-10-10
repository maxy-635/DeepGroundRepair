import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First reduce dimensionality with 1x1 convolution
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Extract features with 3x3 convolution
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Restore dimensionality with another 1x1 convolution
        self.conv3 = nn.Conv2D(in_channels=32, out_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Flatten the output
        self.flatten = nn.Flatten()

        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=7*7, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # MNIST images are 28x28 pixels, so we'll use 28x28
    input_tensor = paddle.to_tensor([[[[1.0]*28]*28]*1], dtype=paddle.float32)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)