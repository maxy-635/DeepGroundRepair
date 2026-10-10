import paddle
import paddle.nn as nn
import paddle.distribution as dist
import numpy as np

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_avg_pool = nn.AvgPool2D(kernel_size=7, stride=7, padding=0)
        self.block1_fc1 = nn.Linear(in_features=64, out_features=64)
        self.block1_relu1 = nn.ReLU()
        self.block1_fc2 = nn.Linear(in_features=64, out_features=64)
        self.block1_relu2 = nn.ReLU()
        self.block1_reshape = dist.ReshapeTransform(in_event_shape='mask', out_event_shape='mask')


        # Block 2
        self.block2_conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.block2_relu1 = nn.ReLU()
        self.block2_conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.block2_relu2 = nn.ReLU()
        self.block2_max_pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Fusion and Classification
        self.fc1 = nn.Linear(in_features=64*8*8, out_features=128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Block 1
        block1_output = self.block1_avg_pool(x)
        block1_output = self.block1_fc1(block1_output)
        block1_output = self.block1_relu1(block1_output)
        block1_output = self.block1_fc2(block1_output)
        block1_output = self.block1_relu2(block1_output)
        block1_output = self.block1_reshape(block1_output)

        # Block 2
        block2_output = self.block2_conv1(x)
        block2_output = self.block2_relu1(block2_output)
        block2_output = self.block2_conv2(block2_output)
        block2_output = self.block2_relu2(block2_output)
        block2_output = self.block2_max_pool(block2_output)

        # Fusion
        fused_output = block1_output + block2_output

        # Classification
        fused_output = paddle.reshape(fused_output, [fused_output.shape[0], -1])
        output = self.fc1(fused_output)
        output = self.relu1(output)
        output = self.fc2(output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)