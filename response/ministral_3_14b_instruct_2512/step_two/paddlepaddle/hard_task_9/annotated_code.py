import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Branch 2: 1x1 conv followed by 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Branch 3: 1x1 conv followed by two 3x3 convs
        self.branch3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # 1x1 conv to adjust dimensions after concatenation
        self.conv1x1 = nn.Conv2D(in_channels=192, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Main path: three fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(in_features=3*32*32, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=10)  # CIFAR-10 has 10 classes
        )

        # Identity connection
        self.identity = nn.Sequential()

    def forward(self, x):
        # Main path
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)

        # Concatenate outputs from all branches
        concat_out = paddle.concat([branch1_out, branch2_out, branch3_out], axis=1)

        # Adjust dimensions with 1x1 conv
        adjusted_out = self.conv1x1(concat_out)

        # Add identity connection
        main_path_out = adjusted_out + x

        # Flatten for fully connected layers
        flattened = paddle.flatten(main_path_out, start_axis=1)

        # Classification output
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, channels, height, width]
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)