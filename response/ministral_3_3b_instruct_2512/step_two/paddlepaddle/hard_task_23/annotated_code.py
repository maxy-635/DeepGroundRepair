import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: Local feature extraction
        self.local_branch = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

        # Branch 2: Downsampling + Upsampling
        self.downsampling_branch = nn.Sequential(
            nn.AvgPool2D(kernel_size=2, stride=2, padding=0),
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2DTranspose(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0)
        )

        # Branch 3: Downsampling + Upsampling
        self.downsampling_branch2 = nn.Sequential(
            nn.AvgPool2D(kernel_size=2, stride=2, padding=0),
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2DTranspose(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0)
        )

        # Concatenation and final processing
        self.concatenate = nn.Sequential(
            nn.Conv2D(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask'),
            nn.Flatten(),
            nn.Linear(in_features=64*8*8, out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Branch 1 output
        local_output = self.local_branch(x)  # output tensor shape: (1, 64, 32, 32)

        # Branch 2 output
        downsampling_output = self.downsampling_branch(x)  # output tensor shape: (1, 32, 32, 32)
        upsampled_output = self.downsampling_branch2(x)  # output tensor shape: (1, 32, 32, 32)

        # Concatenate outputs
        concatenated = paddle.concat([local_output, downsampling_output, upsampled_output], axis=1)  # output tensor shape: (1, 128, 32, 32)
        processed_output = self.concatenate(concatenated)

        return processed_output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # Using CIFAR-10 dimensions: 32x32
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)