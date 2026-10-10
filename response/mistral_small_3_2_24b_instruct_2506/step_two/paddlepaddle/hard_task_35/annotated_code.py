import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()

        # Define the block for each branch
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2D(output_size=1),
            nn.Linear(in_features='mask', out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU()
        )

        # Define the two branches
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Define the final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64 * 2 * 32 * 32, out_features=num_classes)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process through branch1
        branch1_out = self.branch1(x)  # output tensor shape: (1, 64, 32, 32)
        weights1 = self.block(branch1_out)
        weights1 = weights1.reshape(branch1_out.shape)
        branch1_out = branch1_out * weights1

        # Process through branch2
        branch2_out = self.branch2(x)
        weights2 = self.block(branch2_out)
        weights2 = weights2.reshape(branch2_out.shape)
        branch2_out = branch2_out * weights2

        # Concatenate the outputs
        concat_out = paddle.concat([branch1_out, branch2_out], axis=1)

        # Flatten and pass through the final fully connected layer
        flatten_out = self.flatten(concat_out)
        output = self.fc(flatten_out)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)