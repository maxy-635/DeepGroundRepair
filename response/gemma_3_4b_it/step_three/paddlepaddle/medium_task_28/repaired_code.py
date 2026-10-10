import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Attention Layer
        self.attention_conv = nn.Conv2D(
            in_channels=3,  # Assuming CIFAR-10 input is 3 channels
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.attention_softmax = nn.Softmax(axis=-1)

        # Dimensionality Reduction Layer
        self.reduction_conv = nn.Conv2D(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)






        self.layer_norm = nn.LayerNorm(normalized_shape=1)
        self.relu = nn.ReLU()

        # Restore Dimensionality Layer
        self.restore_conv = nn.Conv2D(
            in_channels=1,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Classification Layers
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=3 * 3 * 3, out_features=10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Attention Layer
        attention_weights = self.attention_conv(x)  # output tensor shape: (1, 1, 32, 32)
        attention_weights = self.attention_softmax(attention_weights)  # output tensor shape: (1, 1, 32, 32)
        x = x * attention_weights  # output tensor shape: (1, 3, 32, 32)

        # Dimensionality Reduction Layer
        x = self.reduction_conv(x)
        x = self.layer_norm(x)
        x = self.relu(x)

        # Restore Dimensionality Layer
        x = self.restore_conv(x)

        # Classification Layers
        x = self.flatten(x)
        x = self.linear(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (batch_size, channels, height, width)
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)