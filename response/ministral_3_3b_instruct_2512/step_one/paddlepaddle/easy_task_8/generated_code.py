import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Depthwise separable convolutional layer
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=32)
        self.dropout1 = nn.Dropout(p=0.2)

        # 1x1 convolutional layer
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=1, stride=1)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=64)
        self.dropout2 = nn.Dropout(p=0.2)

        # Flatten and fully connected layer
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=64 * 7 * 7, out_features=10)
        self.activation = nn.CELU(alpha=1.0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer_norm1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.layer_norm2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.linear(x)
        x = self.activation(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.to_tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])
    output_tensor = model(input_tensor)
    print("Input Tensor Shape:", input_tensor.shape)
    print("Output Tensor Shape:", output_tensor.shape)