import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Max pooling layers with different window sizes
        self.pool1 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.pool4 = nn.MaxPool2D(kernel_size=4, stride=4, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=4032, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Apply max pooling with different window sizes
        x1 = self.pool1(x)  # output tensor shape: (1, 3, 32, 32)
        x2 = self.pool2(x)  # output tensor shape: (1, 3, 16, 16)
        x4 = self.pool4(x)  # output tensor shape: (1, 3, 8, 8)

        # Flatten each output
        x1_flat = paddle.flatten(x1, start_axis=1)  # output tensor shape: (1, 3072)
        x2_flat = paddle.flatten(x2, start_axis=1)  # output tensor shape: (1, 768)
        x4_flat = paddle.flatten(x4, start_axis=1)  # output tensor shape: (1, 192)

        # Concatenate the flattened outputs
        combined = paddle.concat([x1_flat, x2_flat, x4_flat], axis=1)  # output tensor shape: (1, 4032)

        # Pass through fully connected layers
        out = self.fc1(combined)
        out = self.fc2(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]