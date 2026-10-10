import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: main path and branch path
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.branch = nn.Sequential(
            nn.Conv2D(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
        )

        # Second block: three max pooling layers
        self.pool1 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.MaxPool2D(kernel_size=2, padding=0)
        self.pool3 = nn.MaxPool2D(kernel_size=4, stride=4, padding=0)

        # Flatten and concatenate outputs
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=1024, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # First block: main path and branch path
        main_path = self.conv1(x)
        main_path = self.conv2(main_path)
        branch_path = self.branch(x)
        output_block1 = paddle.add(main_path, branch_path)

        # Second block: three max pooling layers
        pool1 = self.pool1(output_block1)
        pool2 = self.pool2(pool1)
        pool3 = self.pool3(pool2)

        # Flatten and concatenate outputs
        flattened = self.flatten(pool3)
        # Note: The concatenation should be done by flattening each pooling result separately
        # and then concatenating them. The above code is incorrect. Here's the corrected version:

        # Correct implementation for concatenation
        flattened1 = self.flatten(pool1)
        flattened2 = self.flatten(pool2)
        flattened3 = self.flatten(pool3)
        concatenated = paddle.concat([flattened1, flattened2, flattened3], axis=1)

        # Fully connected layers
        x = paddle.nn.functional.relu(concatenated)
        x = self.fc1(x)
        x = paddle.nn.functional.relu(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 1, 28, 28))  # Example input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)