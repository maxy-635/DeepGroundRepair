import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the three sequential blocks
        block1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=32),
            nn.ReLU()
        )

        block2 = nn.Sequential(
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        block3 = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=128),
            nn.ReLU()
        )

        # Define the parallel branch
        parallel_branch = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        # Combine the outputs
        self.combined_layer = nn.Linear(in_features=128 + 64, out_features=256)
        self.fc1 = nn.Linear(in_features=256, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

        self.block1 = block1
        self.block2 = block2
        self.block3 = block3
        self.parallel_branch = parallel_branch


    def forward(self, x):
        # Process through the sequential blocks
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)

        # Process through the parallel branch
        out_parallel = self.parallel_branch(x)

        # Add the outputs
        combined_output = out3 + out_parallel

        # Flatten the output
        combined_output = paddle.reshape(combined_output, [combined_output.shape[0], -1])

        # Pass through the fully connected layers
        combined_output = self.combined_layer(combined_output)
        combined_output = paddle.nn.ReLU()(combined_output)
        combined_output = self.fc1(combined_output)
        combined_output = paddle.nn.ReLU()(combined_output)
        output = self.fc2(combined_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)