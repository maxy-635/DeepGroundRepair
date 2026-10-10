import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: main path and branch path
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

        # Second block: three max pooling layers with different scales
        self.pool1 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.MaxPool2D(kernel_size=4, stride=4, padding=0)

        # Flatten layers
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=7*7*1 + 14*14*1 + 7*7*1, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # First block
        main_path = self.conv1(x)
        main_path = self.conv2(main_path)
        output_block1 = paddle.add(main_path, x)

        # Second block
        pool1 = self.pool1(output_block1)
        pool2 = self.pool2(output_block1)
        pool3 = self.pool3(output_block1)

        pool1_flat = self.flatten1(pool1)
        pool2_flat = self.flatten2(pool2)
        pool3_flat = self.flatten3(pool3)

        output_block2 = paddle.concat([pool1_flat, pool2_flat, pool3_flat], axis=1)

        # Fully connected layers
        fc1 = self.fc1(output_block2)
        fc1 = paddle.nn.functional.relu(fc1)
        output = self.fc2(fc1)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]