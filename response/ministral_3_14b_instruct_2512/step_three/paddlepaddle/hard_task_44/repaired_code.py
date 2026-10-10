import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three parallel convolutional branches with different kernel sizes
        self.conv1x1_1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3_1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5_1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)

        self.dropout1 = nn.Dropout(p=0.5)

        # Block 2: Four parallel branches with different operations
        # Branch 1: 1x1 conv
        self.conv1x1_2 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Branch 2: 1x1 conv + 3x3 conv
        self.conv1x1_3 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3_3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 3: 1x1 conv + 5x5 conv
        self.conv1x1_4 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv5x5_4 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)

        # Branch 4: 3x3 max pooling + 1x1 conv
        self.maxpool_4 = nn.MaxPool2D(kernel_size=3, stride=1, padding=1)
        self.conv1x1_5 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=262144, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1 processing
        conv1x1_out = self.conv1x1_1(x)  # output tensor shape: (1, 64, 32, 32)
        conv3x3_out = self.conv3x3_1(x)  # output tensor shape: (1, 64, 32, 32)
        conv5x5_out = self.conv5x5_1(x)  # output tensor shape: (1, 64, 32, 32)

        # Concatenate along channel dimension
        block1_out = paddle.concat([conv1x1_out, conv3x3_out, conv5x5_out], axis=1)  # output tensor shape: (1, 192, 32, 32)

        # Apply dropout
        block1_out = self.dropout1(block1_out)  # output tensor shape: (1, 192, 32, 32)

        # Block 2 processing
        # Branch 1
        branch1_out = self.conv1x1_2(block1_out)  # output tensor shape: (1, 64, 32, 32)

        # Branch 2
        branch2_out = self.conv3x3_3(self.conv1x1_3(block1_out))  # output tensor shape: (1, 64, 32, 32)

        # Branch 3
        branch3_out = self.conv5x5_4(self.conv1x1_4(block1_out))  # output tensor shape: (1, 64, 32, 32)

        # Branch 4
        branch4_out = self.conv1x1_5(self.maxpool_4(block1_out))  # output tensor shape: (1, 64, 32, 32)

        # Concatenate all branches
        block2_out = paddle.concat([branch1_out, branch2_out, branch3_out, branch4_out], axis=1)  # output tensor shape: (1, 256, 32, 32)

        # Final layers
        out = self.flatten(block2_out)  # output tensor shape: (1, 262144)
        out = self.fc(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = paddle.randn((1, 3, 32, 32))  # batch_size=1
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)