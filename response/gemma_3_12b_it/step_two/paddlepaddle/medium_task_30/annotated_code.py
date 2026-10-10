import paddle
import paddle.nn.functional as F
import paddle.static.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

    def forward(self, x):
        # Average pooling layers
        x1 = F.avg_pool2d(x, kernel_size=1, stride=1, padding=0)
        x2 = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        x3 = F.avg_pool2d(x, kernel_size=4, stride=4, padding=0)

        # Flatten the outputs of pooling layers
        flatten1 = F.flatten(x1)
        flatten2 = F.flatten(x2)
        flatten3 = F.flatten(x3)

        # Concatenate the flattened features
        merged = paddle.concat([flatten1, flatten2, flatten3], axis=0)

        # Flatten the merged features
        flatten_merged = F.flatten(merged)

        # Fully connected layers
        fc1 = nn.fc(x=flatten_merged, size=128, activation='relu')
        fc2 = nn.fc(x=fc1, size=10)  # 10 classes for CIFAR-10

        return fc2

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.rand(shape=[1, 3, 32, 32])  # Dummy input tensor (batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)