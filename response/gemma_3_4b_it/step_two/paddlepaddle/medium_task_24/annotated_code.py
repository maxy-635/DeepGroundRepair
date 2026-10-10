import paddle
import paddle.nn as nn
import paddle.vision.datasets as datasets

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: <1x1 convolution, 3x3 convolution>
        self.branch1 = nn.Sequential(
            nn.Conv2DLayer(3, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2DLayer(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.5)
        )

        # Branch 2: <1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution>
        self.branch2 = nn.Sequential(
            nn.Conv2DLayer(3, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2DLayer(32, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2DLayer(32, 32, kernel_size=7, stride=1, padding=3),
            nn.Conv2DLayer(32, 32, kernel_size=7, stride=1, padding=3),
            nn.Conv2DLayer(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.5)
        )

        # Branch 3: <max pooling>
        self.branch3 = nn.Conv2DLayer(3, 32, kernel_size=2, stride=2, padding=0)

        # Concatenate the outputs of the branches
        self.concat_layer = nn.Concat(dim=1)

        # Fully connected layers
        self.fc1 = nn.Linear(3 * 32, 64, bias_init='zeros')
        self.fc2 = nn.Linear(64, 10, bias_init='zeros')

    def forward(self, x):
        # Branch 1
        branch1_output = self.branch1(x)

        # Branch 2
        branch2_output = self.branch2(x)

        # Branch 3
        branch3_output = self.branch3(x)

        # Concatenate the outputs
        concatenated_output = self.concat_layer([branch1_output, branch2_output, branch3_output])

        # Flatten the concatenated output
        flattened_output = concatenated_output.reshape([-1, 3 * 32])

        # Fully connected layers
        fc1_output = self.fc1(flattened_output)
        fc2_output = self.fc2(fc1_output)

        return fc2_output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)