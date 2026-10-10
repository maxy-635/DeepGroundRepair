import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Primary path and branch path
        self.primary_path = nn.Sequential(
            nn.Conv2D(1, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2DSeparable(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(64, 128, kernel_size=1, stride=1, padding=0)
        )

        self.branch_path = nn.Sequential(
            nn.Conv2DSeparable(1, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(32, 64, kernel_size=1, stride=1, padding=0)
        )

        # Concatenate the outputs of both paths along the channel dimension
        self.concat = nn.Sequential()

        # Block 2: Channel shuffling
        self.channel_shuffle = nn.Sequential(
            nn.Flatten(),
            nn.ReshapeTransform((None, None, 4, -1)),
            nn.Permute((0, 1, 3, 2)),
            nn.ReshapeTransform((None, None, -1))
        )

        # Fully connected layer for classification
        self.fc = nn.Linear(128, 10)  # Assuming MNIST has 10 classes

    def forward(self, x):
        # Block 1: Primary path and branch path
        primary_output = self.primary_path(x)
        branch_output = self.branch_path(x)

        # Concatenate the outputs
        concatenated = paddle.concat([primary_output, branch_output], axis=1)

        # Block 2: Channel shuffling
        reshaped = self.channel_shuffle(concatenated)

        # Fully connected layer
        output = self.fc(reshaped)
        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = paddle.randn((1, 1, 28, 28))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print(output_tensor.shape)