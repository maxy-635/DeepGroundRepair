import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)

        # Branch path (identity mapping)
        self.branch = nn.Sequential()

        # Final layers
        self.fc3 = nn.Linear(in_features=64, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=10)  # Assuming CIFAR-10 has 10 classes

    def forward(self, x):
        # Main path
        pooled = self.global_avg_pool(x)
        pooled = paddle.flatten(pooled, start_axis=1, stop_axis=-1)
        weights = self.fc1(pooled)
        weights = self.fc2(weights)
        weights = paddle.reshape(weights, shape=[-1, 64, 1, 1])
        main_out = paddle.multiply(x=x, y=weights)

        # Branch path
        branch_out = self.branch(x)

        # Combine paths
        combined = paddle.add(x=main_out, y=branch_out)

        # Final layers
        out = self.fc3(combined)
        out = self.fc4(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 64, 32, 32])  # Assuming input shape is (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]