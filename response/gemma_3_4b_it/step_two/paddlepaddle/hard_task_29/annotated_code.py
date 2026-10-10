import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.main_path = nn.Sequential(
            nn.Conv2D(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(32, 16, kernel_size=3, stride=1, padding=1)
        )
        self.branch_path = nn.Conv2D(1, 16, kernel_size=3, stride=1, padding=1)
        self.merge = nn.Add()

        # Block 2
        self.pool1 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.pool3 = nn.MaxPool2D(kernel_size=4, stride=4, padding=0)

        self.flatten = nn.Flatten()
        self.concat = nn.Concat(dim=1)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        # Block 1
        main_output = self.main_path(x)
        branch_output = self.branch_path(x)
        merged_output = self.merge(main_output, branch_output)

        # Block 2
        pooled_output = self.pool1(merged_output)
        pooled_output = self.pool2(pooled_output)
        pooled_output = self.pool3(pooled_output)
        flattened_output = self.flatten(pooled_output)
        concatenated_output = self.concat(flattened_output, [pooled_output])

        # Fully connected layers
        fc1_output = self.fc1(concatenated_output)
        fc2_output = self.fc2(fc1_output)
        output = self.softmax(fc2_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)