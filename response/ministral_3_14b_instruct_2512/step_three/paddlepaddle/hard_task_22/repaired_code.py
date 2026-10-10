import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path - three groups with different kernel sizes
        self.main_path = nn.LayerList()

        # Group 1: 1x1 convolution
        self.main_path.append(nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        ))

        # Group 2: 3x3 convolution
        self.main_path.append(nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        ))

        # Group 3: 5x5 convolution
        self.main_path.append(nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        ))

        # Branch path - 1x1 convolution
        self.branch_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=192, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=192*32*32, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        # Main path processing
        main_outputs = []
        for layer in self.main_path:
            main_outputs.append(layer(x))

        # Concatenate along channel dimension
        main_output = paddle.concat(main_outputs, axis=1)

        # Branch path processing
        branch_output = self.branch_path(x)

        # Fuse outputs by addition
        fused_output = main_output + branch_output

        # Fully connected layers
        output = self.fc(fused_output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, channels, height, width]
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)