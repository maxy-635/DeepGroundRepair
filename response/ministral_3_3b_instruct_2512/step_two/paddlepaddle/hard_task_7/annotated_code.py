import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolutional layer with 32 kernels
        self.initial_conv = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Block 1
        self.block1_conv1 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.block1_depthwise_conv = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32)
        self.block1_conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Block 2
        self.permutation = nn.LayerParamAttr(name='mask', value='mask')
        self.channel_shuffle = nn.LayerParamAttr(name='channel_shuffle', value=[4, 2])

    def forward(self, x):
        # Initial convolutional layer
        x = self.initial_conv(x)

        # Block 1
        x1 = self.block1_conv1(x)
        x1 = self.block1_depthwise_conv(x1)
        x1 = self.block1_conv2(x1)

        # Split the input into two groups along the last dimension
        x1_split = paddle.slice(x1, axes=[3], starts=[0], ends=[16])
        x2 = paddle.slice(x1, axes=[3], starts=[16], ends=[32])

        # Concatenate the outputs
        x = paddle.concat([x1_split, x2], axis=3)

        # Block 2
        # Obtain the shape of input and reshape the input
        input_shape = x.shape
        height, width, channels = input_shape[1], input_shape[2], input_shape[3]

        # Reshape into four groups
        reshaped_x = paddle.reshape(x, [input_shape[0], height, width, 2, channels // 2])

        # Swap the third and fourth dimensions
        reshaped_x = paddle.permute(reshaped_x, self.permutation)

        # Reshape back to original shape for channel shuffling
        reshaped_x = paddle.reshape(reshaped_x, [input_shape[0], height, width, channels])

        # Flatten the output
        flattened_x = paddle.reshape(reshaped_x, [input_shape[0], -1])

        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=flattened_x.shape[1], out_features=10)
        output = self.fc(flattened_x)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.zeros([1, 1, 28, 28])  # Dummy input tensor for MNIST
    output_tensor = model(input_tensor)
    print(output_tensor.shape)