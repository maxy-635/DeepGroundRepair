import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: parallel branches
        self.branch1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.branch4 = nn.MaxPool2D(kernel_size=3, stride=1, padding=1)

        # Second block: dimensionality reduction and attention mechanism
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc_final = nn.Linear(in_features=64, out_features=10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        # First block: parallel branches
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        branch4_out = self.branch4(x)

        # Concatenate the outputs of the parallel branches
        concat_out = paddle.concat([branch1_out, branch2_out, branch3_out, branch4_out], axis=1)

        # Second block: attention mechanism
        pooled = self.global_avg_pool(concat_out)
        pooled = paddle.reshape(pooled, shape=[-1, 256])  # 4 branches * 64 channels = 256 channels

        # Generate weights
        weights = self.fc1(pooled)
        weights = F.relu(weights)
        weights = self.fc2(weights)
        weights = paddle.reshape(weights, shape=[-1, 256, 1, 1])  # Reshape to match input channels

        # Apply attention
        attention_out = paddle.multiply(concat_out, weights)

        # Final classification
        out = self.fc_final(attention_out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # CIFAR-10 image size is 32x32 with 3 channels
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)