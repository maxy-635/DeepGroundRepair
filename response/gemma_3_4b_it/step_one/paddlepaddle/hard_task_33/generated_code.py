import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2D(1, 32, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2D(32, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2D(16, 16, 1, 1, 0)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2D(1, 32, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2D(32, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2D(16, 16, 1, 1, 0)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2D(1, 32, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2D(32, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2D(16, 16, 1, 1, 0)
        )

        self.concat = nn.Concat(dim=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 3, 10)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        output1 = self.branch1(x)
        output2 = self.branch2(x)
        output3 = self.branch3(x)

        concatenated_output = self.concat([output1, output2, output3])
        flattened_output = self.flatten(concatenated_output)
        classification_output = self.fc(flattened_output)
        probabilities = self.softmax(classification_output)

        return probabilities


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(1, 1, 28, 28)  # Example input shape: batch_size, channels, height, width
    output_tensor = model(input_tensor)
    print(output_tensor.shape)