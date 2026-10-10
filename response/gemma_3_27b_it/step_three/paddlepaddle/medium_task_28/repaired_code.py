import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Attention Mechanism
        self.attention_conv = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.attention_softmax = nn.Softmax(axis=1)

        # Dimensionality Reduction and Restoration
        self.reduction_conv = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.layer_norm = nn.LayerNorm(normalized_shape=[1, 1, 32, 32])
        self.relu = nn.ReLU()
        self.restoration_conv = nn.Conv2D(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Classification Head
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=3072, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Attention Mechanism
        attention_weights = self.attention_softmax(self.attention_conv(x))  # output tensor shape: (1, 1, 32, 32)
        contextual_features = paddle.multiply(x, attention_weights)  # output tensor shape: (1, 3, 32, 32)

        # Dimensionality Reduction and Restoration
        reduced_features = self.reduction_conv(x)  # output tensor shape: (1, 1, 32, 32)
        normalized_features = self.layer_norm(reduced_features)
        activated_features = self.relu(normalized_features)
        restored_features = self.restoration_conv(activated_features)
        
        # Residual Connection
        processed_output = x + restored_features

        # Classification Head
        flattened_output = self.flatten(processed_output)
        classification_results = self.linear(flattened_output)

        return classification_results


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)