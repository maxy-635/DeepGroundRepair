import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the three max pooling layers with different window sizes
        self.pool1 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.MaxPool2D(kernel_size=4, stride=4, padding=0)

        # Flatten layers for each pooling output
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()

        # Concatenate the flattened features
        self.concat = nn.Sequential()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=0, out_features=128)  # Placeholder, actual in_features will be set later
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Apply the three max pooling layers
        pool1_out = self.pool1(x)
        pool2_out = self.pool2(x)
        pool3_out = self.pool3(x)

        # Flatten each pooling output
        flat1 = self.flatten1(pool1_out)
        flat2 = self.flatten2(pool2_out)
        flat3 = self.flatten3(pool3_out)

        # Concatenate the flattened features
        concat_out = paddle.concat([flat1, flat2, flat3], axis=1)

        # Fully connected layers
        fc1_out = self.fc1(concat_out)
        fc2_out = self.fc2(fc1_out)

        return fc2_out

if __name__ == "__main__":
    model = DL_Model()

    # Create a dummy input tensor with shape [batch_size, channels, height, width]
    # For CIFAR-10, typical input shape is [batch_size, 3, 32, 32]
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])

    # Forward pass to get output
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)