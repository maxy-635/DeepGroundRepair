import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Attention mechanism
        self.attention_conv = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(axis=1)

        # Feature processing
        self.reduce_conv = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.layer_norm = nn.LayerNorm(normalized_shape=[1, 1, 32, 32])
        self.relu = nn.ReLU()
        self.restore_conv = nn.Conv2D(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Classification
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=3 * 32 * 32, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Attention mechanism
        attention_weights = self.softmax(self.attention_conv(x))  # output tensor shape: (1, 1, 32, 32)
        contextual_features = x * attention_weights  # output tensor shape: (1, 3, 32, 32)

        # Feature processing
        reduced_features = self.reduce_conv(x)  # output tensor shape: (1, 1, 32, 32)
        normalized_features = self.layer_norm(reduced_features)
        activated_features = self.relu(normalized_features)
        restored_features = self.restore_conv(activated_features)

        # Residual connection
        processed_output = paddle.add(x, restored_features)

        # Classification
        flattened_output = self.flatten(processed_output)
        classification_output = self.linear(flattened_output)

        return classification_output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)