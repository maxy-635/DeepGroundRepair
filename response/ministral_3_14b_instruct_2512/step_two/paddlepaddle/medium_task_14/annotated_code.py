import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Sequential blocks with Conv2D, BatchNorm2D, and ReLU
        self.block1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels='mask', kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=128),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=256),
            nn.ReLU()
        )

        # Parallel branch
        self.parallel_branch = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=256*4*4, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

        # Pooling layer for downsampling
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process through sequential blocks
        block1_out = self.block1(x)  # output tensor shape of this layer should be (1, 128, 32, 32)
        block2_out = self.block2(block1_out)  # output tensor shape: (1, 128, 32, 32)
        block3_out = self.block3(block2_out)  # output tensor shape: (1, 256, 32, 32)

        # Process through parallel branch
        parallel_out = self.parallel_branch(x)  # output tensor shape: (1, 64, 32, 32)

        # Add all outputs together
        combined = block1_out + block2_out + block3_out + parallel_out

        # Apply pooling and flatten
        pooled = self.pool(combined)
        flattened = pooled.flatten(start_axis=1)

        # Fully connected layers
        fc1_out = self.fc1(flattened)
        fc2_out = self.fc2(fc1_out)

        return fc2_out

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, 3, 32, 32] for CIFAR-10
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)