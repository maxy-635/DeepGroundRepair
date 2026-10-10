import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main pathway: 1x1 convolution
        self.conv1x1 = nn.Conv2D(3, 64, kernel_size=1, stride=1, padding=0)

        # Parallel branch: 1x1, 1x3, and 3x1 convolutions
        self.conv_parallel_branch = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(64, 64, kernel_size=1, stride=1, padding=0)
        )

        # Concatenate outputs and apply 1x1 convolution
        self.conv_main = nn.Conv2D(192, 64, kernel_size=1, stride=1, padding=0)

        # Direct connection from input
        self.skip_connection = nn.Conv2D(3, 64, kernel_size=1, stride=1, padding=0)

        # Fully connected layers for classification
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Main pathway
        main_pathway = self.conv1x1(x)

        # Parallel branch
        parallel_branch = self.conv_parallel_branch(x)

        # Concatenate outputs
        concatenated = paddle.concat([main_pathway, parallel_branch], axis=1)

        # Apply 1x1 convolution
        main_output = self.conv_main(concatenated)

        # Direct connection
        skip_connection_output = self.skip_connection(x)

        # Additive fusion
        fused_output = main_output + skip_connection_output

        # Flatten and pass through fully connected layers
        x = self.flatten(fused_output)
        x = paddle.nn.functional.relu(self.fc1(x))
        x = paddle.nn.functional.softmax(self.fc2(x), axis=1)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Example input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)