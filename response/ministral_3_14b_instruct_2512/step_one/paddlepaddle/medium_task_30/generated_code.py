import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the three average pooling layers with different window sizes and strides
        self.pool1 = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.pool2 = nn.AdaptiveAvgPool2D(output_size=(2, 2))
        self.pool3 = nn.AdaptiveAvgPool2D(output_size=(4, 4))

        # Flatten layers for each pooling output
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()

        # Concatenate the flattened features
        self.concat = paddle.concat

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=3 * 32 * 32, out_features=512)  # Assuming input channels=32
        self.fc2 = nn.Linear(in_features=512, out_features=10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        # Apply the three average pooling layers
        pool1_out = self.pool1(x)
        pool2_out = self.pool2(x)
        pool3_out = self.pool3(x)

        # Flatten each pooling output
        flat1 = self.flatten1(pool1_out)
        flat2 = self.flatten2(pool2_out)
        flat3 = self.flatten3(pool3_out)

        # Concatenate the flattened features
        concat_out = self.concat([flat1, flat2, flat3], axis=1)

        # Fully connected layers
        fc1_out = self.fc1(concat_out)
        fc2_out = self.fc2(fc1_out)

        return fc2_out

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, channels, height, width]
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)