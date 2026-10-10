import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

        # Branch path (direct connection)
        self.branch = nn.Identity()

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=28*28, out_features=10)  # MNIST images are 28x28

    def forward(self, x):
        # Main path
        main_path = self.conv1(x)
        main_path = self.relu(main_path)
        main_path = self.conv2(main_path)

        # Branch path (direct connection)
        branch_path = self.branch(x)

        # Combine paths through addition
        combined = main_path + branch_path

        # Final layers
        combined = self.flatten(combined)
        output = self.fc(combined)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, channels, height, width]
    # MNIST images are 28x28 grayscale (1 channel)
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)