import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First 1x1 convolutional layer
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=32)
        self.dropout1 = nn.Dropout(p=0.5)

        # Second 1x1 convolutional layer
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=32)
        self.dropout2 = nn.Dropout(p=00.5)

        # 3x1 convolutional layer
        self.conv3 = nn.Conv2D(in_channels=32, out_features=32, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.layer_norm3 = nn.LayerNorm(normalized_shape=32)
        self.dropout3 = nn.Dropout(p=0.5)

        # 1x3 convolutional layer
        self.conv4 = nn.Conv2D(in_channels=32, out_features=1, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.layer_norm4 = nn.LayerNorm(normalized_shape=1)
        self.dropout4 = nn.Dropout(p=0.5)

        # Restore channels to match input
        self.conv5 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.layer_norm5 = nn.LayerNorm(normalized_shape=32)
        self.dropout5 = nn.Dropout(p=0.5)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=784, out_features=10)

    def forward(self, x):
        # First 1x1 convolution
        x = self.conv1(x)
        x = self.layer_norm1(x)
        x = self.dropout1(x)

        # Second 1x1 convolution
        x = self.conv2(x)
        x = self.layer_norm2(x)
        x = self.dropout2(x)

        # 3x1 convolution
        x = self.conv3(x)
        x = self.layer_norm3(x)
        x = self.dropout3(x)

        # 1x3 convolution
        x = self.conv4(x)
        x = self.layer_norm4(x)
        x = self.dropout4(x)

        # Restore channels
        x_restore = self.conv5(x)
        x_restore = self.layer_norm5(x_restore)
        x_restore = self.dropout5(x_restore)

        # Add original input with restored features
        x = paddle.add(x, x_restore)

        # Flatten and fully connected
        x = self.flatten(x)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = paddle.randn((1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)