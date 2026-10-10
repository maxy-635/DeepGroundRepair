import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the convolutional blocks for each group
        self.conv_block = nn.Sequential(
            nn.Conv2D(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask'),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Main path
        self.main_path = nn.Sequential(
            self.conv_block,
            self.conv_block,
            self.conv_block
        )

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=32 * 32 * 32, out_features=10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split the input into three groups along the channel dimension
        group1 = x[:, 0:1, :, :]  # output tensor shape: (1, 1, 32, 32)
        group2 = x[:, 1:2, :, :]  # output tensor shape: (1, 1, 32, 32)
        group3 = x[:, 2:3, :, :]  # output tensor shape: (1, 1, 32, 32)

        # Process each group through the convolutional blocks
        out_group1 = self.conv_block(group1)
        out_group2 = self.conv_block(group2)
        out_group3 = self.conv_block(group3)

        # Combine the outputs from the three groups
        main_path = paddle.add(out_group1, paddle.add(out_group2, out_group3))

        # Fuse the main path with the original input
        combined_features = paddle.add(main_path, x)

        # Flatten the combined features
        flattened_features = self.flatten(combined_features)

        # Pass the flattened features through the fully connected layer
        output = self.fc(flattened_features)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor with shape (1, 3, 32, 32) - batch size 1, 3 channels, 32x32 image
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)