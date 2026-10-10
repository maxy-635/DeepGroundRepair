import paddle
import paddle.nn as nn
import paddle.vision.transforms as transforms

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Channel splitting and feature extraction
        self.conv1_1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv1_3 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1_5 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)

        # Concatenation
        self.concat = nn.Concat(axis=1)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=32 * 32 * 32, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=10)

        # Activation and normalization
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(normalized_shape=[512])
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        # Channel splitting
        channel1 = self.conv1_1(x)
        channel2 = self.conv1_3(x)
        channel3 = self.conv1_5(x)

        # Feature extraction and ReLU activation
        channel1 = self.relu(channel1)
        channel2 = self.relu(channel2)
        channel3 = self.relu(channel3)

        # Concatenation
        x = self.concat([channel1, channel2, channel3])

        # Flatten
        x = paddle.flatten(x, start_axis=1, end_axis=-1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.layernorm(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        # Softmax activation
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)