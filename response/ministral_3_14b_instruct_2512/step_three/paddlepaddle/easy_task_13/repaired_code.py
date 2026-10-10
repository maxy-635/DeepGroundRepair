import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.nn.initializer as init

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the convolutional layers
        self.conv1x1_1 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv1x1_2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.conv1x3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv1x1_restore = nn.Conv2D(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Define dropout layers
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)
        self.dropout4 = nn.Dropout(p=0.5)

        # Define flatten and linear layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=784, out_features=10)  # MNIST has 784 features (28x28)

        # Initialize weights using Xavier uniform
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                layer.weight.set_value(init.XavierUniform()(layer.weight.shape))
                if layer.bias is not None:
                    layer.bias.set_value(init.Constant(value=0.0)(layer.bias.shape))
            elif isinstance(layer, nn.Linear):
                layer.weight.set_value(init.XavierUniform()(layer.weight.shape))
                if layer.bias is not None:
                    layer.bias.set_value(init.Constant(value=0.0)(layer.bias.shape))

    def forward(self, x):
        # First 1x1 convolution
        x1 = self.conv1x1_1(x)
        x1 = self.dropout1(x1)

        # Second 1x1 convolution
        x2 = self.conv1x1_2(x1)
        x2 = self.dropout2(x2)

        # 3x1 convolution
        x3 = self.conv3x1(x2)
        x3 = self.dropout3(x3)

        # 1x3 convolution
        x4 = self.conv1x3(x3)
        x4 = self.dropout4(x4)

        # Restore channels to match input
        x5 = self.conv1x1_restore(x4)

        # Add the processed features to the original input
        x = x + x5

        # Flatten and fully connected layer
        x = self.flatten(x)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 1, 28, 28))  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)