import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Split the input into three groups along the last dimension
        # Each group undergoes feature extraction via depthwise separable convolutional layers
        # with distinct kernel sizes of 1x1, 3x3, and 5x5

        # Group 1: 1x1 kernel
        self.group1_conv = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(num_features=3),
            nn.ReLU(),
            nn.LPPool2D(norm_type=2, kernel_size=2, stride=2)
        )

        # Group 2: 3x3 kernel
        self.group2_conv = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=3),
            nn.ReLU(),
            nn.LPPool2D(norm_type=2, kernel_size=2, stride=2)
        )

        # Group 3: 5x5 kernel
        self.group3_conv = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2D(num_features=3),
            nn.ReLU(),
            nn.LPPool2D(norm_type=2, kernel_size=2, stride=2)
        )

        # Concatenate the outputs of the three groups
        self.concat = nn.Sequential()

        # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features='mask', out_features=10),
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split the input into three groups along the last dimension
        # Since the input is already in NCHW format, we need to split the channels
        # Here, we assume the input has 3 channels and we split them into three groups of 1 channel each
        # But since the input is 3 channels, we need to adjust the approach
        # Instead, we'll process the input directly with the three groups in parallel

        # Process each group
        group1_out = self.group1_conv(x)  # output tensor shape: (1, 3, 16, 16)
        group2_out = self.group2_conv(x)  # output tensor shape: (1, 3, 16, 16)
        group3_out = self.group3_conv(x)  # output tensor shape: (1, 3, 16, 16)

        # Concatenate along the channel dimension
        concat_out = paddle.concat([group1_out, group2_out, group3_out], axis=1)  # output tensor shape: (1, 9, 16, 16)

        # Fully connected layer
        fc_out = self.fc(concat_out)

        return fc_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)