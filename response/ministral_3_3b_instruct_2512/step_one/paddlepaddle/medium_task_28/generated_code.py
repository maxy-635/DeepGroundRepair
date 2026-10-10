import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Step 1: Generate attention weights with 1x1 convolution followed by softmax
        self.attention_conv = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(axis=1)

        # Step 2: Reduce dimensionality to one-third with 1x1 convolution, followed by layer normalization and ReLU
        self.reduce_conv = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.layer_norm = nn.LayerNorm(normalized_shape=3)
        self.reLU = nn.ReLU()

        # Step 3: Restore dimensionality with another 1x1 convolution
        self.restore_conv = nn.Conv2D(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Step 4: Flatten and fully connected layer for classification
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=3*32*32, out_features=10)

    def forward(self, x):
        # Step 1: Generate attention weights
        attention_weights = self.attention_conv(x)
        attention_weights = self.softmax(attention_weights)

        # Step 2: Reduce dimensionality
        reduced_features = self.reduce_conv(x)
        reduced_features = self.layer_norm(reduced_features)
        reduced_features = self.reLU(reduced_features)

        # Step 3: Restore dimensionality
        restored_features = self.restore_conv(reduced_features)

        # Step 4: Add original input and processed features
        processed_output = x + restored_features

        # Flatten and classify
        flattened = self.flatten(processed_output)
        logits = self.fc(flattened)

        return logits

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = paddle.randn((1, 3, 32, 32))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input tensor shape:", input_tensor.shape)
    print("Output tensor shape:", output_tensor.shape)