import paddle
import paddle.nn as nn
import paddle.vision.datasets as datasets
import paddle.io

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1 = nn.Layer(
            paddle.nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            paddle.nn.Dropout(p=0.1),
            paddle.nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            paddle.nn.Dropout(p=0.1)
        )

        # Block 2
        self.block2 = nn.Layer(
            paddle.nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
            paddle.nn.Dropout(p=0.1),
            paddle.nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            paddle.nn.Dropout(p=0.1),
            paddle.nn.Conv2D(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            paddle.nn.Dropout(p=0.1)
        )

        # Output layer
        self.flatten = paddle.nn.Flatten()
        self.fc = paddle.nn.Linear(in_features=32 * 8 * 8, out_features=10)

    def forward(self, x):
        # Block 1
        x = self.block1(x)

        # Block 2
        x = self.block2(x)

        # Output layer
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # Instantiate the model
    model = DL_Model()

    # Create a dummy input tensor
    input_tensor = paddle.randn(batch_size=1, channels=3, height=32, width=32)

    # Pass the input through the model
    output_tensor = model(input_tensor)

    # Print the output shape
    print("Output shape:", output_tensor.shape)