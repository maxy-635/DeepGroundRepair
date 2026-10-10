import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Three parallel paths with max pooling of different scales
        self.pooling_path1 = nn.MaxPool2D(kernel_size=1, stride=1)
        self.pooling_path2 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.pooling_path3 = nn.MaxPool2D(kernel_size=4, stride=4)

        # Dropout layers for regularization
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)

        # Flatten layers for each path
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()

        # Concatenate the flattened outputs
        self.concat = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            paddle.reshape  # Use paddle.reshape instead of nn.Reshape
        )

        # Block 2: Four parallel paths with different convolution and pooling strategies
        # Path 1: Single 1x1 convolution
        self.conv_path1 = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1),
            nn.MaxPool2D(kernel_size=2, stride=2)
        )

        # Path 2: 1x1, 1x7, and 7x1 convolutions
        self.conv_path2 = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=(1, 7), stride=1),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=(7, 1), stride=1),
            nn.MaxPool2D(kernel_size=2, stride=2)
        )

        # Path 3: Alternating 7x1 and 1x7 convolutions
        self.conv_path3 = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=(7, 1), stride=1),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=(1, 7), stride=1),
            nn.MaxPool2D(kernel_size=2, stride=2)
        )

        # Path 4: Average pooling with 1x1 convolution
        self.conv_path4 = nn.Sequential(
            nn.AdaptiveAvgPool2D(output_size=(1, 1)),
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1)
        )

        # Concatenate the outputs of all paths along the channel dimension
        self.concat_block2 = nn.Sequential(
            paddle.reshape,  # Use paddle.reshape for reshaping
            nn.Linear(32 * 4, 1024)
        )

        # Final classification layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Block 1 processing
        path1 = self.pooling_path1(x)
        path2 = self.pooling_path2(x)
        path3 = self.pooling_path3(x)

        # Flatten and apply dropout
        flattened1 = self.flatten1(path1)
        flattened2 = self.flatten2(path2)
        flattened3 = self.flatten3(path3)

        dropout1 = self.dropout1(flattened1)
        dropout2 = self.dropout2(flattened2)
        dropout3 = self.dropout3(flattened3)

        # Concatenate flattened outputs
        concatenated = paddle.concat([dropout1, dropout2, dropout3], axis=1)

        # Reshape and pass through fully connected layer
        reshaped = paddle.reshape(concatenated, shape=(-1, 1024))

        # Block 2 processing
        path1_block2 = self.conv_path1(reshaped)
        path2_block2 = self.conv_path2(reshaped)
        path3_block2 = self.conv_path3(reshaped)
        path4_block2 = self.conv_path4(reshaped)

        # Concatenate outputs along channel dimension
        concatenated_block2 = paddle.concat([path1_block2, path2_block2, path3_block2, path4_block2], axis=1)

        # Reshape and pass through fully connected layers
        reshaped_block2 = paddle.reshape(concatenated_block2, shape=(-1, 32 * 4))
        reshaped_block2 = self.concat_block2(reshaped_block2)

        # Final classification
        x = self.fc1(reshaped_block2)
        x = paddle.nn.functional.relu(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = paddle.randn((1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print("Input tensor shape:", input_tensor.shape)
    print("Output tensor shape:", output_tensor.shape)