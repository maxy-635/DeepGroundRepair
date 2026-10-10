import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.avg_pool = nn.AvgPool2D(kernel_size=7, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=21632, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 32, 32, 32)
        # Global average pooling
        x = self.avg_pool(x)  # output tensor shape: (1, 32, 26, 26)
        # Flatten the output
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)  # output tensor shape: (1, 21632)
        # First fully connected layer
        x = self.fc1(x)
        # Second fully connected layer
        weights = self.fc2(x)
        # Reshape weights to match input channels
        weights = paddle.reshape(weights, shape=[0, 32, 1, 1])
        # Element-wise multiplication
        x = paddle.multiply(x=x, y=weights)
        # Flatten and final fully connected layer
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 32, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)