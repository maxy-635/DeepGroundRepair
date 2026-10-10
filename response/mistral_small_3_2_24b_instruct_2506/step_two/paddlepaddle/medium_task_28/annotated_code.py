import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Attention mechanism
        self.attention_conv = nn.Conv2D(
            in_channels=3,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.softmax = nn.Softmax(axis=1)

        # Dimensionality reduction
        self.reduce_conv = nn.Conv2D(
            in_channels=3,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=[3, 32, 32])
        self.relu = nn.ReLU()

        # Dimensionality restoration
        self.restore_conv = nn.Conv2D(
            in_channels=1,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Classification head
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=3*32*32, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Generate attention weights
        attention_weights = self.softmax(self.attention_conv(x))  # output tensor shape: (1, 3, 32, 32)

        # Apply attention
        contextual_features = x * attention_weights  # output tensor shape: (1, 3, 32, 32)

        # Reduce dimensionality
        reduced_features = self.relu(self.layer_norm(self.reduce_conv(contextual_features)))

        # Restore dimensionality
        restored_features = self.restore_conv(reduced_features)

        # Residual connection
        output = x + restored_features

        # Classification
        output = self.flatten(output)
        output = self.fc(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected shape: [1, 10]