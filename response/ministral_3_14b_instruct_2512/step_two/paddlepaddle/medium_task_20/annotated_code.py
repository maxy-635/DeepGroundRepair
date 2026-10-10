import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Path 2: 1x1 + two 3x3 convolutions
        self.path2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Path 3: 1x1 + single 3x3 convolution
        self.path3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Path 4: max pooling + 1x1 convolution
        self.path4 = nn.Sequential(
            nn.MaxPool2D(kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Final layers
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features='mask', out_features=128)
        self.output = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process each path
        path1_out = self.path1(x)  # output tensor shape: (1, 64, 32, 32)
        path2_out = self.path2(x)  # output tensor shape: (1, 64, 32, 32)
        path3_out = self.path3(x)  # output tensor shape: (1, 64, 32, 32)
        path4_out = self.path4(x)  # output tensor shape: (1, 64, 32, 32)

        # Concatenate all paths
        concat = paddle.concat([path1_out, path2_out, path3_out, path4_out], axis=1)  # output tensor shape: (1, 256, 32, 32)

        # Flatten and pass through dense layers
        flat = self.flatten(concat)  # output tensor shape: (1, 262144)
        dense_out = self.dense(flat)
        output = self.output(dense_out)

        # Apply softmax
        return F.softmax(output, axis=1)

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, 3, 32, 32] (CIFAR-10 image size)
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)