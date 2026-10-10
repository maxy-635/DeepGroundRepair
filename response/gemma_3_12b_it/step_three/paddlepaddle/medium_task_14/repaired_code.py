import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=128),
            nn.ReLU()
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=256),
            nn.ReLU()
        )

        # Parallel branch
        self.parallel_conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.parallel_conv2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.parallel_conv3 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)


        # Fully connected layers
        self.fc1 = nn.Linear(in_features=256 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Path 1: Sequential blocks
        out1 = self.block1(x)  # output tensor shape: (1, 64, 32, 32)
        out2 = self.block2(out1)  # output tensor shape: (1, 128, 32, 32)
        out3 = self.block3(out2)  # output tensor shape: (1, 256, 32, 32)

        # Path 2: Parallel branch
        out_parallel1 = self.parallel_conv1(x)  # output tensor shape: (1, 64, 32, 32)
        out_parallel2 = self.parallel_conv2(out_parallel1)  # output tensor shape: (1, 128, 32, 32)
        out_parallel3 = self.parallel_conv3(out_parallel2)  # output tensor shape: (1, 256, 32, 32)

        # Add the outputs from both paths
        added_output = paddle.add(out3, out_parallel3)  # output tensor shape: (1, 256, 32, 32)

        # Flatten the output
        flattened_output = nn.Flatten()(added_output)  # output tensor shape: (1, 262144)

        # Fully connected layers
        fc1_output = self.fc1(flattened_output)
        output = self.fc2(fc1_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for CIFAR-10 (32x32 images, 3 channels)
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # Batch size of 1
    output_tensor = model(input_tensor)
    print(output_tensor.shape)