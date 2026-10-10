import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.fc1 = nn.Linear(in_features=64, out_features=64)  # Assuming input channels are 64
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)  # CIFAR-10 has 10 classes

        # Branch path
        self.conv_branch = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Final fully connected layers
        self.fc_final1 = nn.Linear(in_features=64, out_features=128)
        self.fc_final2 = nn.Linear(in_features=128, out_features=64)
        self.fc_final3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # Main path
        pooled = self.global_avg_pool(x)
        pooled = paddle.reshape(pooled, shape=[-1, 64])  # Flatten to [batch_size, channels]
        weights = self.fc1(pooled)
        weights = self.fc2(weights)
        weights = paddle.reshape(weights, shape=[-1, 64, 1, 1])  # Reshape to match input channels
        main_output = paddle.multiply(x=x, y=weights)

        # Branch path
        branch_output = self.conv_branch(x)

        # Combine outputs
        combined = paddle.add(x=main_output, y=branch_output)

        # Final fully connected layers
        combined = paddle.reshape(combined, shape=[-1, 64])  # Flatten
        combined = self.fc_final1(combined)
        combined = self.fc_final2(combined)
        output = self.fc_final3(combined)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 64, 32, 32])  # Batch size 1, 64 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]