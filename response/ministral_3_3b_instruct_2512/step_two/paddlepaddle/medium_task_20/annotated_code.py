import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the four parallel convolutional paths
        self.path1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.path2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.path3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.path4 = nn.Sequential(
            nn.MaxPool2D(kernel_size='mask', stride='mask', padding='mask'),
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Concatenate the outputs of the four paths
        self.concatenate = nn.Flatten()
        self.dense = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),  # Assuming input size is 32x32, after pooling it's 16x16, then 8x8 after concatenation
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        # Final output layer
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Forward pass through each path
        path1_out = self.path1(x)  # output tensor shape: (1, 64, 32, 32)
        path2_out = self.path2(x)  # output tensor shape: (1, 64, 32, 32)
        path3_out = self.path3(x)  # output tensor shape: (1, 64, 32, 32)
        path4_out = self.path4(x)  # output tensor shape of this layer should be (1, 64, 32, 32)

        # Concatenate the outputs
        concatenated = paddle.concat([path1_out, path2_out, path3_out, path4_out], axis=1)
        flattened = self.concatenate(concatenated)

        # Pass through dense layer
        dense_out = self.dense(flattened)
        output = self.output(dense_out)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)