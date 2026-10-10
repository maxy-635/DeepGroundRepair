import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()
        # Main path
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.fc1 = nn.Linear(in_features=3072, out_features=1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=1024, out_features=3072)

        # Branch path (identity)

        # Fully connected layers for final classification
        self.fc3 = nn.Linear(in_features=6144, out_features=1024)
        self.relu2 = nn.ReLU()
        self.fc4 = nn.Linear(in_features=1024, out_features=10)

    def forward(self, x):
        # Main path
        main_path = self.adaptive_avg_pool(x)
        main_path = nn.Flatten()(main_path)
        main_path = self.fc1(main_path)
        main_path = self.relu1(main_path)
        main_path = self.fc2(main_path)
        main_path = main_path.reshape(x.shape[0], 3072, 1, 1)  # Reshape to match input shape
        main_path = paddle.multiply(x, main_path)

        # Branch path
        branch_path = x

        # Combine paths
        combined_path = paddle.add(main_path, branch_path)

        # Flatten the combined path
        combined_path = combined_path.reshape(combined_path.shape[0], -1)

        # Final classification layers
        x = self.fc3(combined_path)
        x = self.relu2(x)
        x = self.fc4(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))  # Example input tensor (batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)