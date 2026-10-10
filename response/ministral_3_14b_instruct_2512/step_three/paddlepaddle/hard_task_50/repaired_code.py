import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: three max pooling layers with different scales
        self.max_pool_1x1 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.max_pool_2x2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.max_pool_4x4 = nn.MaxPool2D(kernel_size=4, stride=4, padding=0)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

        # Fully connected layer for first block
        self.fc1 = nn.Linear(in_features=4032, out_features=1024)

        # Second block: separable convolutions with different kernel sizes
        self.sep_conv_1x1 = nn.Conv2D(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0, groups=1024)
        self.sep_conv_3x3 = nn.Conv2D(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1, groups=1024)
        self.sep_conv_5x5 = nn.Conv2D(in_channels=1024, out_channels=256, kernel_size=5, stride=1, padding=2, groups=1024)
        self.sep_conv_7x7 = nn.Conv2D(in_channels=1024, out_channels=256, kernel_size=7, stride=1, padding=3, groups=1024)

        # Final fully connected layer
        self.fc2 = nn.Linear(in_features=256 * 4 * 8 * 8, out_features=10)  # Assuming output shape after convs

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First block processing
        pool1 = self.max_pool_1x1(x)  # output tensor shape: (1, 3, 32, 32)
        pool2 = self.max_pool_2x2(x)  # output tensor shape: (1, 3, 16, 16)
        pool3 = self.max_pool_4x4(x)  # output tensor shape: (1, 3, 8, 8)

        # Flatten each pooling output
        pool1_flat = paddle.flatten(pool1, start_axis=1, stop_axis=-1)  # output tensor shape: (1, 3072)
        pool2_flat = paddle.flatten(pool2, start_axis=1, stop_axis=-1)  # output tensor shape: (1, 768)
        pool3_flat = paddle.flatten(pool3, start_axis=1, stop_axis=-1)  # output tensor shape: (1, 192)

        # Concatenate and apply dropout
        concat = paddle.concat([pool1_flat, pool2_flat, pool3_flat], axis=1)  # output tensor shape: (1, 4032)
        dropout_out = self.dropout(concat)  # output tensor shape: (1, 4032)

        # Fully connected layer and reshape
        fc_out = self.fc1(dropout_out)
        fc_out = paddle.reshape_(fc_out, shape=[-1, 1024, 1, 1])

        # Second block processing
        # Split into four groups along the last dimension
        split_out = paddle.split(fc_out, num_or_sections=4, axis=3)

        # Process each group with separable convolutions
        conv1 = self.sep_conv_1x1(split_out[0])
        conv2 = self.sep_conv_3x3(split_out[1])
        conv3 = self.sep_conv_5x5(split_out[2])
        conv4 = self.sep_conv_7x7(split_out[3])

        # Concatenate the outputs
        concat_conv = paddle.concat([conv1, conv2, conv3, conv4], axis=1)

        # Flatten and final fully connected layer
        flat_conv = paddle.flatten(concat_conv, start_axis=1, stop_axis=-1)
        output = self.fc2(flat_conv)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)