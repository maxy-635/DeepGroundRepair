import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Average Pooling Layer
        self.avg_pool = nn.AvgPool2D(kernel_size=5, stride=3, padding=0)

        # 1x1 Convolutional Layer
        self.conv_1x1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Flatten Layer
        self.flatten = nn.Flatten()

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=32 * 14 * 14, out_features=128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Average Pooling
        x = self.avg_pool(x)

        # 1x1 Convolutional Layer
        x = self.conv_1x1(x)

        # Flatten
        x = self.flatten(x)

        # Fully Connected Layers
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = paddle.randn((1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print("Input Tensor Shape:", input_tensor.shape)
    print("Output Tensor Shape:", output_tensor.shape)