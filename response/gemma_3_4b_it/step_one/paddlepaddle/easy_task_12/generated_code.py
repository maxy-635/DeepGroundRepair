import paddle
import paddle.vision.datasets as datasets
import paddle.io as pdo
import paddle.nn as nn

class DL_Model(paddle.nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2D(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2D(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Match dimensions
        self.match_layer = nn.Conv2D(16, 32, kernel_size=1, stride=1, padding=0)

        # Flatten and fully connected layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, 10)  # Adjust output size based on input image size

    def forward(self, x):
        main_output = self.main_path(x)
        branch_output = self.branch_path(x)

        matched_output = self.match_layer(branch_output)

        combined_output = main_output + matched_output

        flattened_output = self.flatten(combined_output)
        output = self.fc(flattened_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(1, 1, 28, 28, requires_grad=False)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)