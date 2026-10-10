import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Max Pooling Layers
        self.max_pool1 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.max_pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.max_pool3 = nn.MaxPool2D(kernel_size=4, stride=4, padding=0)
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()

        # Fully connected layer and reshape before Block 2
        self.fc1 = nn.Linear(in_features=1, out_features=1024)
        self.reshape = nn.Reshape((1, 1024, 1, 1)) # Reshape to 4D tensor

        # Block 2: Convolutional and Max Pooling Branches
        self.conv1x1 = nn.Conv2D(in_channels=1024, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(in_channels=1024, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2D(in_channels=1024, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.max_pool_block2 = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.flatten_block2 = nn.Flatten()
        self.fc2 = nn.Linear(in_features=64*1*1, out_features=10)  # Assuming 10 classes for MNIST


    def forward(self, x):
        # Block 1
        x1 = self.max_pool1(x)
        x2 = self.max_pool2(x)
        x3 = self.max_pool3(x)

        flat_x1 = self.flatten1(x1)
        flat_x2 = self.flatten2(x2)
        flat_x3 = self.flatten3(x3)

        concat_x = paddle.concat([flat_x1, flat_x2, flat_x3], axis=-1)

        # Fully connected layer and reshape
        fc_out = self.fc1(concat_x)
        reshaped_x = self.reshape(fc_out)

        # Block 2
        conv1x1_out = self.conv1x1(reshaped_x)
        conv3x3_out = self.conv3x3(reshaped_x)
        conv5x5_out = self.conv5x5(reshaped_x)
        max_pool_out = self.max_pool_block2(reshaped_x)

        concat_block2 = paddle.concat([conv1x1_out, conv3x3_out, conv5x5_out, max_pool_out], axis=1)
        flatten_block2_out = self.flatten_block2(concat_block2)
        output = self.fc2(flatten_block2_out)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for MNIST (1 image, 1 channel, 28x28)
    input_tensor = paddle.rand(shape=[1, 1, 28, 28], dtype='float32')
    output_tensor = model(input_tensor)
    print(output_tensor.shape)