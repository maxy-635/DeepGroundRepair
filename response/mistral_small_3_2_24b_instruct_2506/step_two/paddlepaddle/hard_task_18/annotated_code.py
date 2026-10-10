import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: feature extraction
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)

        # Second block: channel attention
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc_classifier = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # First block
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        pooled_out = self.avg_pool(conv2_out)

        # Second block
        global_avg_out = self.global_avg_pool(pooled_out)
        global_avg_out = paddle.reshape(global_avg_out, shape=[-1, 64])
        fc1_out = self.fc1(global_avg_out)
        fc2_out = self.fc2(fc1_out)
        fc2_out = paddle.reshape(fc2_out, shape=[-1, 64, 1, 1])
        channel_weights = paddle.multiply(pooled_out, fc2_out)

        # Flatten and classify
        flattened = paddle.reshape(channel_weights, shape=[-1, 64])
        output = self.fc_classifier(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # CIFAR-10 input shape: (batch_size, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]