import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Three average pooling layers with different kernel sizes and strides
        self.avg_pool1 = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.avg_pool2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.avg_pool3 = nn.AvgPool2D(kernel_size=4, stride=4, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=4032, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Apply three average pooling layers
        x1 = self.avg_pool1(x)  # output tensor shape: (1, 3, 32, 32)
        x2 = self.avg_pool2(x)  # output tensor shape: (1, 3, 16, 16)
        x3 = self.avg_pool3(x)  # output tensor shape: (1, 3, 8, 8)

        # Flatten each pooling output
        x1_flat = paddle.flatten(x1, start_axis=1)  # output tensor shape: (1, 3072)
        x2_flat = paddle.flatten(x2, start_axis=1)  # output tensor shape: (1, 768)
        x3_flat = paddle.flatten(x3, start_axis=1)  # output tensor shape: (1, 192)

        # Concatenate the flattened features
        x_concat = paddle.concat([x1_flat, x2_flat, x3_flat], axis=1)  # output tensor shape: (1, 4032)

        # Further flatten and process through fully connected layers
        x_flat = paddle.flatten(x_concat, start_axis=1)  # output tensor shape: (1, 4032)
        x_fc1 = F.relu(self.fc1(x_flat))
        x_fc2 = self.fc2(x_fc1)

        return x_fc2

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print(output_tensor)