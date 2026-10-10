import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: three average pooling layers with different scales
        self.avg_pool1 = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.avg_pool2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.avg_pool3 = nn.AvgPool2D(kernel_size=4, stride=4, padding=0)

        # Flatten and concatenate the pooled features
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()

        # Fully connected layer and reshape for second block
        self.fc1 = nn.Linear(28*28 + 14*14 + 7*7, 28*28)  # Sum of flattened dimensions
        self.reshape = paddle.distribution.ReshapeTransform(in_event_shape='mask', out_event_shape='mask')




        # Second block: depthwise separable convolutions with different kernel sizes
        self.conv1x1 = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=1, num_channels=1, epsilon=1e-5),
            nn.ReLU()
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=1, epsilon=1e-5),
            nn.ReLU()
        )

        self.conv5x5 = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(num_groups=1, num_channels=1, epsilon=1e-5),
            nn.ReLU()
        )

        self.conv7x7 = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3),
            nn.GroupNorm(num_groups=1, num_channels=1, epsilon=1e-5),
            nn.ReLU()
        )

        # Final classification layer
        self.fc2 = nn.Linear(4*28*28, 10)  # 4 groups concatenated

    def forward(self, x):
        # the initial shape of x is: (32, 1, 28, 28)
        # First block processing
        pool1 = self.avg_pool1(x)  # output tensor shape: (32, 1, 28, 28)
        pool2 = self.avg_pool2(x)  # output tensor shape: (32, 1, 14, 14)
        pool3 = self.avg_pool3(x)  # output tensor shape: (32, 1, 7, 7)

        # Flatten and concatenate
        flat1 = self.flatten1(pool1)  # output tensor shape: (32, 784)
        flat2 = self.flatten2(pool2)  # output tensor shape: (32, 196)
        flat3 = self.flatten3(pool3)  # output tensor shape: (32, 49)

        # Concatenate along channel dimension
        concat = paddle.concat([flat1, flat2, flat3], axis=1)  # output tensor shape: (32, 1029)

        # Fully connected and reshape
        fc_out = self.fc1(concat)  # output tensor shape: (32, 784)
        reshaped = self.reshape(fc_out)

        # Second block processing
        # Split into 4 groups along channel dimension
        group1 = reshaped[:, :1, :, :]
        group2 = reshaped[:, 1:2, :, :]
        group3 = reshaped[:, 2:3, :, :]
        group4 = reshaped[:, 3:4, :, :]

        # Process each group with different convolutions
        conv1 = self.conv1x1(group1)
        conv2 = self.conv3x3(group2)
        conv3 = self.conv5x5(group3)
        conv4 = self.conv7x7(group4)

        # Concatenate along channel dimension
        concat_conv = paddle.concat([conv1, conv2, conv3, conv4], axis=1)

        # Flatten and final classification
        flat_final = paddle.flatten(concat_conv, start_axis=1)
        output = self.fc2(flat_final)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([32, 1, 28, 28])  # Batch of 32 MNIST images (1 channel, 28x28)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)