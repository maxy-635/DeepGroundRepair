import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Get input shape dynamically (assuming input is 3D for CIFAR-10)
        # We'll use a placeholder for the input shape during initialization
        # The actual input shape will be determined when the model is called
        self.fc = nn.Linear(in_features=1024, out_features=10)  # Assuming flattened size of 1024 for CIFAR-10

    def forward(self, input):
        # Get input shape
        input_shape = paddle.shape(input)

        # Reshape input to (height, width, groups, channels_per_group)
        # For CIFAR-10, input shape is (batch_size, channels, height, width)
        # We need to permute dimensions to (batch_size, height, width, channels)
        input_permuted = paddle.transpose(input, perm=[0, 2, 3, 1])

        # Reshape to (batch_size, height, width, groups, channels_per_group)
        # For CIFAR-10, channels=3, groups=3, channels_per_group=1
        channels = input_shape[1]
        groups = 3
        channels_per_group = channels // groups
        reshaped = paddle.reshape(input_permuted,
                                 shape=[input_shape[0], input_shape[2], input_shape[3], groups, channels_per_group])

        # Swap third and fourth dimensions
        permuted = paddle.transpose(reshaped, perm=[0, 1, 2, 4, 3])

        # Reshape back to original input shape
        output_shape = [input_shape[0], input_shape[1], input_shape[2], input_shape[3]]
        output = paddle.reshape(permuted, shape=output_shape)

        # Flatten the output for the fully connected layer
        flattened = paddle.reshape(output, shape=[-1, 1024])

        # Pass through fully connected layer with softmax
        fc_out = self.fc(flattened)
        softmax_out = F.softmax(fc_out, axis=1)

        return softmax_out

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)