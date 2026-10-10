import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path with two feature extraction branches
        self.conv1x1_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.conv_sequence_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0)
        )

        # Branch that connects directly to input
        self.branch = nn.Sequential()

        # Final 1x1 convolution to align dimensions
        self.final_conv = nn.Conv2D(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')

        # Classification layers
        self.fc1 = nn.Linear(in_features=3*32*32, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path processing
        path1_out = self.conv1x1_path(x)  # output tensor shape: (1, 64, 32, 32)
        path2_out = self.conv_sequence_path(x)  # output tensor shape: (1, 3, 32, 32)

        # Concatenate the two paths
        concat_out = paddle.concat([path1_out, path2_out], axis=1)  # output tensor shape: (1, 67, 32, 32)

        # Apply final 1x1 convolution
        main_path_out = self.final_conv(concat_out)

        # Add the branch (which is just the input in this case)
        branch_out = x
        output = main_path_out + branch_out

        # Flatten for classification
        output = paddle.flatten(output, start_axis=1)

        # Classification layers
        output = self.fc1(output)
        output = F.relu(output)
        output = self.fc2(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, channels, height, width]
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)