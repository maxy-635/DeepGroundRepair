import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three max pooling layers with different scales
        self.max_pool_1x1 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.max_pool_2x2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.max_pool_4x4 = nn.MaxPool2D(kernel_size=4, stride=4, padding=0)

        # Flatten and concatenate the outputs
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=1, out_features=1)  # Placeholder, actual in_features will be determined by input shape

        # Reshape for Block 2
        self.reshape_transform = paddle.distribution.ReshapeTransform(
            in_event_shape=[1, 1, 1, 1],  # Placeholder, actual shape will be determined by input shape
            out_event_shape=[1, 1, 1, 1]   # Placeholder, actual shape will be determined by input shape
        )

        # Block 2: Multiple branches
        self.conv1x1 = nn.Conv2D(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2D(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.max_pool_3x3 = nn.MaxPool2D(kernel_size=3, stride=1, padding=1)

        # Final classification layers
        self.fc2 = nn.Linear(in_features=1, out_features=10)  # Placeholder, actual in_features will be determined by input shape

    def forward(self, x):
        # Block 1
        pool1 = self.max_pool_1x1(x)
        pool2 = self.max_pool_2x2(x)
        pool3 = self.max_pool_4x4(x)

        # Flatten and concatenate
        pool1_flat = self.flatten(pool1)
        pool2_flat = self.flatten(pool2)
        pool3_flat = self.flatten(pool3)

        # Concatenate along axis 1
        concat = paddle.concat([pool1_flat, pool2_flat, pool3_flat], axis=1)

        # Fully connected layer and reshape
        fc_output = self.fc1(concat)
        reshaped = self.reshape_transform(fc_output)

        # Block 2
        conv1x1_out = self.conv1x1(reshaped)
        conv3x3_out = self.conv3x3(reshaped)
        conv5x5_out = self.conv5x5(reshaped)
        max_pool_3x3_out = self.max_pool_3x3(reshaped)

        # Concatenate all branch outputs
        concat_branches = paddle.concat([conv1x1_out, conv3x3_out, conv5x5_out, max_pool_3x3_out], axis=1)

        # Final classification
        flat = self.flatten(concat_branches)
        output = self.fc2(flat)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # MNIST images are 28x28 with 1 channel
    input_tensor = paddle.randn([1, 1, 28, 28])  # Batch size 1, 1 channel, 28x28
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)