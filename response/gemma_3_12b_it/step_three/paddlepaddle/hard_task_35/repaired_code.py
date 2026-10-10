import paddle
import paddle.nn as nn
import paddle.static as static

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the block
        self.block = nn.Sequential(
            nn.AvgPool2D(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(in_features=32*32*3, out_features=128, name="fc1"),
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=256, name="fc2")
        )

        # Define the two branches
        self.branch1 = self.block
        self.branch2 = self.block

        # Concatenate the outputs of the two branches
        self.concat = nn.Concat(axis=1)

        # Flatten the concatenated output
        self.flatten = nn.Flatten()

        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=256*2, out_features=10, name="fc_out")

    def forward(self, x):
        # Pass the input through the two branches
        out1 = self.branch1(x)
        out2 = self.branch2(x)

        # Concatenate the outputs of the two branches
        out_concat = self.concat([out1, out2])

        # Flatten the concatenated output
        out_flatten = self.flatten(out_concat)

        # Pass the flattened output through the fully connected layer
        output = self.fc(out_flatten)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # Batch size 1, 3 channels, 32x32 image

    # Execute the model
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)