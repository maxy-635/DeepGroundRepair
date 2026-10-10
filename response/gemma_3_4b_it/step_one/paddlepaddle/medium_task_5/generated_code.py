import paddle
import paddle.nn as nn
import paddle.vision.datasets as datasets
import paddle.io as io

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)

        # Branch path
        self.conv3 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2D(kernel_size=2, stride=2)

        # Combine paths
        self.add = nn.Add()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)  # Adjust input size based on pooling
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        # Main path
        main_path = self.conv1(x)
        main_path = self.pool1(main_path)
        main_path = self.conv2(main_path)
        main_path = self.pool2(main_path)

        # Branch path
        branch_path = self.conv3(x)
        branch_path = self.pool3(branch_path)

        # Combine paths
        combined = self.add(main_path, branch_path)

        # Flatten and fully connected layers
        flattened = self.flatten(combined)
        output = self.fc1(flattened)
        output = self.fc2(output)
        output = self.softmax(output)

        return output


if __name__ == "__main__":
    # Create an instance of the DL_Model
    model = DL_Model()

    # Define a dummy input tensor
    input_tensor = paddle.randn(batch_size=1, channels=3, height=32, width=32)  # Example input shape

    # Pass the input tensor through the model
    output_tensor = model(input_tensor)

    # Print the output shape to verify
    print("Output tensor shape:", output_tensor.shape)