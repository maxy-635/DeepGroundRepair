import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the layers
        self.conv = nn.Conv2D(
            in_channels=3,
            out_channels=32,
            kernel_size=7,
            stride=1,
            padding=3,
            groups=1,
            bias_attr=False
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=[32])
        self.fc1 = nn.Linear(
            in_features=32 * 5 * 5,
            out_features=32
        )
        self.fc2 = nn.Linear(
            in_features=32,
            out_features=10
        )

    def forward(self, x):
        # Extract spatial features
        x = self.conv(x)
        x = self.layer_norm(x)
        x = F.relu(x)

        # Flatten the features
        x = paddle.flatten(x, [1, 2])

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        # Apply softmax activation
        x = F.softmax(x, axis=-1)
        return x

if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = paddle.rand(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)