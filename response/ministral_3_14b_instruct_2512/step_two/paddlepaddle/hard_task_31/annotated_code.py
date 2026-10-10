import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: main path and branch path
        self.main_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.2),
            nn.Conv2D(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0)
        )

        # Second block: three separable convolution groups
        self.group1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.2)
        )

        self.group2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.2)
        )

        self.group3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.Dropout(p=0.2)
        )

        # Final classification layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features='mask', out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First block processing
        main_output = self.main_path(x)  # output tensor shape: (1, 3, 32, 32)
        branch_output = x  # output tensor shape: (1, 3, 32, 32)
        block1_output = main_output + branch_output  # output tensor shape: (1, 3, 32, 32)

        # Second block processing
        group1_output = self.group1(block1_output)  # output tensor shape: (1, 32, 32, 32)
        group2_output = self.group2(block1_output)  # output tensor shape: (1, 32, 32, 32)
        group3_output = self.group3(block1_output)  # output tensor shape: (1, 32, 32, 32)

        # Concatenate along channel dimension
        block2_output = paddle.concat([group1_output, group2_output, group3_output], axis=1)  # output tensor shape: (1, 96, 32, 32)

        # Final classification
        features = self.flatten(block2_output)  # output tensor shape: (1, 98304)
        output = self.fc(features)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, channels, height, width]
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)