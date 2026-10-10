import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Global Average Pooling and Fully Connected Layers
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.fc1 = nn.Linear(32 * 32, 1024)  # Assuming input channels are 32*32 for CIFAR-10
        self.fc2 = nn.Linear(1024, 1024)
        self.layer_norm = nn.LayerNorm(1024)

        # Block 2: Convolutional Layers and Max Pooling
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2D(kernel_size=2, stride=2)

        # Branch from Block 1 to Block 2
        self.fc_branch = nn.Linear(1024, 128 * 1 * 1)  # Reshape to match Block 2's output shape

        # Fully Connected Layers for Classification
        self.fc_classify1 = nn.Linear(256, 512)
        self.fc_classify2 = nn.Linear(512, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1: Global Average Pooling and Fully Connected Layers
        x_pool = self.global_avg_pool(x)  # output tensor shape: (1, 3, 1, 1)
        x_pool = paddle.reshape(x_pool, shape=[-1, 32 * 32])
        x_fc1 = self.fc1(x_pool)
        x_fc1 = self.fc2(x_fc1)
        x_fc1 = self.layer_norm(x_fc1)

        # Branch from Block 1 to Block 2
        branch_output = self.fc_branch(x_fc1)
        branch_output = paddle.reshape(branch_output, shape=[-1, 128, 1, 1])

        # Block 2: Convolutional Layers and Max Pooling
        x_conv1 = self.conv1(x)
        x_conv1 = paddle.nn.functional.relu(x_conv1)
        x_conv2 = self.conv2(x_conv1)
        x_conv2 = paddle.nn.functional.relu(x_conv2)
        x_pool2 = self.max_pool(x_conv2)

        # Fuse the outputs from Block 1 and Block 2
        fused_output = x_conv2 + branch_output

        # Fully Connected Layers for Classification
        x_flatten = paddle.reshape(fused_output, shape=[-1, 128 * 1 * 1])
        x_classify1 = self.fc_classify1(x_flatten)
        x_classify1 = paddle.nn.functional.relu(x_classify1)
        x_classify2 = self.fc_classify2(x_classify1)

        return x_classify2

if __name__ == "__main__":
    model = DL_Model()
    # Assuming input shape for CIFAR-10 is (batch_size, channels, height, width)
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print("Input Tensor Shape:", input_tensor.shape)
    print("Output Tensor Shape:", output_tensor.shape)