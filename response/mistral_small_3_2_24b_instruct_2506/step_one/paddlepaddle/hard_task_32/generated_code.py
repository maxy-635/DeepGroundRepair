import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the specialized block for each branch
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.5),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.5)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.5),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.5)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.5),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.5)
        )

        # Concatenate the outputs from the three branches
        self.concat = nn.Sequential(
            nn.Linear(in_features=32*3, out_features=128),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        # Process each branch
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)

        # Concatenate the outputs
        out = paddle.concat([out1, out2, out3], axis=1)

        # Flatten the concatenated output
        out = paddle.flatten(out, start_axis=1, stop_axis=-1)

        # Process through fully connected layers
        out = self.concat(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]