import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Feature extraction at multiple scales
        self.conv1x1_1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3_1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5_1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)

        self.maxpool1 = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        # Second set of convolutions
        self.conv1x1_2 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3_2 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5_2 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=5, stride=1, padding=2)

        # Third set of convolutions
        self.conv1x1_3 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3_3 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5_3 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=5, stride=1, padding=2)

        # Final convolution before flattening
        self.final_conv = nn.Conv2D(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=128*4*4, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

        # Activation functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First set of convolutions
        conv1x1_out = self.conv1x1_1(x)  # output tensor shape: (1, 64, 32, 32)
        conv3x3_out = self.conv3x3_1(x)  # output tensor shape: (1, 64, 32, 32)
        conv5x5_out = self.conv5x5_1(x)  # output tensor shape: (1, 64, 32, 32)

        # Concatenate features
        concat1 = paddle.concat([conv1x1_out, conv3x3_out, conv5x5_out], axis=1)  # output tensor shape: (1, 192, 32, 32)

        # Max pooling
        pool1 = self.maxpool1(concat1)  # output tensor shape: (1, 192, 16, 16)

        # Second set of convolutions
        conv1x1_out2 = self.conv1x1_2(pool1)  # output tensor shape: (1, 64, 16, 16)
        conv3x3_out2 = self.conv3x3_2(pool1)  # output tensor shape: (1, 64, 16, 16)
        conv5x5_out2 = self.conv5x5_2(pool1)  # output tensor shape: (1, 64, 16, 16)

        # Concatenate features
        concat2 = paddle.concat([conv1x1_out2, conv3x3_out2, conv5x5_out2], axis=1)  # output tensor shape: (1, 192, 16, 16)

        # Third set of convolutions
        conv1x1_out3 = self.conv1x1_3(concat2)  # output tensor shape: (1, 64, 16, 16)
        conv3x3_out3 = self.conv3x3_3(concat2)  # output tensor shape: (1, 64, 16, 16)
        conv5x5_out3 = self.conv5x5_3(concat2)  # output tensor shape: (1, 64, 16, 16)

        # Concatenate features
        concat3 = paddle.concat([conv1x1_out3, conv3x3_out3, conv5x5_out3], axis=1)  # output tensor shape: (1, 192, 16, 16)

        # Final convolution
        final_conv_out = self.final_conv(concat3)  # output tensor shape: (1, 128, 16, 16)

        # Flatten and fully connected layers
        flat = self.flatten(final_conv_out)  # output tensor shape: (1, 32768)
        fc1_out = self.relu(self.fc1(flat))
        fc1_out = self.dropout(fc1_out)
        fc2_out = self.fc2(fc1_out)

        return fc2_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Should be [1, 10] for CIFAR-10 classes