import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: three max pooling layers with different scales
        self.pool1 = nn.MaxPool2D(kernel_size=(1, 1), stride=(1, 1))
        self.pool2 = nn.MaxPool2D(kernel_size=(2, 2), stride=(2, 2))
        self.pool3 = nn.MaxPool2D(kernel_size=(4, 4), stride=(4, 4))

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.5)

        # Flatten layers
        self.flatten1 = nn.Flatten()

        # Fully connected layer and reshape
        self.fc = nn.Linear(in_features=1024, out_features=4)
        self.reshape = paddle.distribution.ReshapeTransform(in_event_shape=(1, 3, 32, 32), out_event_shape=(1, 1024))

    def forward(self, x):
        # First block processing
        x1 = self.flatten1(self.pool1(x))
        x2 = self.flatten1(self.pool2(x))
        x3 = self.flatten1(self.pool3(x))

        # Concatenate and apply dropout
        x_concat = paddle.concat([x1, x2, x3], axis=1)
        x_concat = self.dropout(x_concat)

        # Reshape and pass through fully connected layer
        x_reshape = self.reshape(x_concat)
        x_fc = self.fc(x_reshape)

        return x_fc

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = paddle.randn((1, 3, 32, 32))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)