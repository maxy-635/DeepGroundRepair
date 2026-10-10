import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First 1x1 convolutional layer
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.dropout1 = nn.Dropout(p=0.5)

        # Second 1x1 convolutional layer
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.dropout2 = nn.Dropout(p=0.5)

        # 3x1 convolutional layer
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=1, padding=0)
        self.dropout3 = nn.Dropout(p=0.5)

        # 1x3 convolutional layer
        self.conv4 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding=0)
        self.dropout4 = nn.Dropout(p=0.5)

        # Restore channels to input's channel count (1)
        self.conv5 = nn.Conv2D(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.dropout5 = nn.Dropout(p=0.5)

        # Flatten and fully connected layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=28*28, out_features=10)

    def forward(self, x):
        # Save original input
        original_input = x

        # First 1x1 convolution
        x = self.conv1(x)
        x = self.dropout1(x)

        # Second 1x1 convolution
        x = self.conv2(x)
        x = self.dropout2(x)

        # 3x1 convolution
        x = self.conv3(x)
        x = self.dropout3(x)

        # 1x3 convolution
        x = self.conv4(x)
        x = self.dropout4(x)

        # Restore channels
        x = self.conv5(x)
        x = self.dropout5(x)

        # Add original input
        x = x + original_input

        # Flatten and fully connected
        x = self.flatten(x)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 1, 28, 28))  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Should print [1, 10]