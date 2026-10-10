import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.fc = nn.Linear(3072, 10)  # Assuming input is 32x32x3 and output is 10 classes

    def forward(self, x):
        # the initial shape of x is: (1, 32, 32, 3)
        # Get the shape of the input tensor
        shape = paddle.shape(x)  # output tensor shape: (4)

        # Reshape the input tensor into three groups
        x = paddle.reshape(x, shape=[-1, 3, 32, 32])

        # Swap the third and fourth dimensions
        x = paddle.transpose(x, perm=[0, 1, 2, 4, 3])

        # Reshape back to original input shape
        x = paddle.reshape(x, shape=[shape[0], shape[1], shape[2], shape[3]])

        # Flatten the tensor for the fully connected layer
        x = paddle.flatten(x, start_axis=1)

        # Pass through the fully connected layer with softmax activation
        x = self.fc(x)
        x = F.softmax(x, axis=1)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 32, 32, 3])  # Batch size of 1, 32x32 image with 3 channels
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]