import paddle
import paddle.vision.datasets as datasets
import paddle.nn.functional as F
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1
        self.branch1 = nn.Layer(
            nn.Conv2D(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2D(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # Branch 2
        self.branch2 = nn.Layer(
            nn.Conv2D(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2D(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # Branch 3
        self.branch3 = nn.Layer(
            nn.Conv2D(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2D(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )

        # Concatenate the outputs of the three branches
        self.concat = nn.Concat([self.branch1.output, self.branch2.output, self.branch3.output], dim=1)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # Adjust the input size based on your convolutional output size
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax2D()

    def forward(self, x):
        x = x.astype('float32')
        x = F.normalize(x, p=2, axis=1)
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        branch3_output = self.branch3(x)
        concatenated = self.concat(branch1_output, branch2_output, branch3_output)
        fc1_output = self.relu1(self.fc1(concatenated))
        fc1_output = self.dropout1(fc1_output)
        output = self.fc2(fc1_output)
        output = self.softmax(output)
        return output


if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with the expected shape
    input_tensor = paddle.randn(1, 1, 28, 28, requires_grad=False)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor)