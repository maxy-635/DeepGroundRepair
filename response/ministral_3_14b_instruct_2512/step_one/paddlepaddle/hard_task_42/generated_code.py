import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three parallel max pooling paths
        self.max_pool1 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.max_pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.max_pool3 = nn.MaxPool2D(kernel_size=4, stride=4, padding=0)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

        # Fully connected layer and reshaping between blocks
        self.fc1 = nn.Linear(28*28*1 + 28*28*1 + 7*7*1, 28*28*4)  # Input channels: 1+1+1, output channels: 4
        self.reshape = nn.Flatten()

        # Block 2: Four parallel convolutional paths
        # Path 1: Single 1x1 convolution
        self.conv1_1 = nn.Conv2D(in_channels=4, out_channels=16, kernel_size=1, stride=1, padding=0)

        # Path 2: 1x1 + 1x7 + 7x1 convolutions
        self.conv2_1 = nn.Conv2D(in_channels=4, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=(1,7), stride=1, padding=(0,3))
        self.conv2_3 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=(7,1), stride=1, padding=(3,0))

        # Path 3: 1x1 + 7x1 + 1x7 convolutions
        self.conv3_1 = nn.Conv2D(in_channels=4, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv3_2 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=(7,1), stride=1, padding=(3,0))
        self.conv3_3 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=(1,7), stride=1, padding=(0,3))

        # Path 4: Average pooling + 1x1 convolution
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=(1,1))
        self.conv4 = nn.Conv2D(in_channels=4, out_channels=16, kernel_size=1, stride=1, padding=0)

        # Final classification layers
        self.fc2 = nn.Linear(16*4*4, 128)  # Assuming output size after block 2 is 4x4
        self.fc3 = nn.Linear(128, 10)

        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        # Block 1 processing
        # Path 1: 1x1 max pooling
        pool1 = self.max_pool1(x)
        pool1 = self.relu(pool1)
        pool1 = self.reshape(pool1)
        pool1 = self.dropout1(pool1)

        # Path 2: 2x2 max pooling
        pool2 = self.max_pool2(x)
        pool2 = self.relu(pool2)
        pool2 = self.reshape(pool2)
        pool2 = self.dropout2(pool2)

        # Path 3: 4x4 max pooling
        pool3 = self.max_pool3(x)
        pool3 = self.relu(pool3)
        pool3 = self.reshape(pool3)
        pool3 = self.dropout3(pool3)

        # Concatenate all paths
        concat = paddle.concat([pool1, pool2, pool3], axis=1)

        # Reshape and fully connected layer
        fc_input = self.fc1(concat)
        fc_input = self.reshape(fc_input)
        fc_input = fc_input.reshape((-1, 4, 28, 28))  # Reshape to 4 channels

        # Block 2 processing
        # Path 1: Single 1x1 convolution
        path1 = self.conv1_1(fc_input)
        path1 = self.relu(path1)

        # Path 2: 1x1 + 1x7 + 7x1 convolutions
        path2 = self.conv2_1(fc_input)
        path2 = self.relu(path2)
        path2 = self.conv2_2(path2)
        path2 = self.relu(path2)
        path2 = self.conv2_3(path2)
        path2 = self.relu(path2)

        # Path 3: 1x1 + 7x1 + 1x7 convolutions
        path3 = self.conv3_1(fc_input)
        path3 = self.relu(path3)
        path3 = self.conv3_2(path3)
        path3 = self.relu(path3)
        path3 = self.conv3_3(path3)
        path3 = self.relu(path3)

        # Path 4: Average pooling + 1x1 convolution
        path4 = self.avg_pool(fc_input)
        path4 = self.conv4(path4)
        path4 = self.relu(path4)

        # Concatenate all paths along channel dimension
        concat_block2 = paddle.concat([path1, path2, path3, path4], axis=1)

        # Final classification
        flatten = self.reshape(concat_block2)
        fc2 = self.fc2(flatten)
        fc2 = self.relu(fc2)
        output = self.fc3(fc2)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, 1, 28, 28] for MNIST
    input_tensor = paddle.randn((32, 1, 28, 28))  # batch_size=32
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)