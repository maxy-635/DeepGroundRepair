import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Feature extraction branches
        self.branch1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

        # Concatenation and dimension adjustment
        self.concat = nn.Concat(axis=1)
        self.conv_concat = nn.Conv2D(in_channels=64 * 3, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Direct connection (identity mapping)
        self.identity = nn.Identity()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=3 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Feature extraction
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)

        # Concatenation and dimension adjustment
        concat_out = self.concat([branch1_out, branch2_out, branch3_out])
        concat_out = self.conv_concat(concat_out)

        # Fusion with identity mapping
        x = concat_out + self.identity(x)

        # Flatten for fully connected layers
        x = paddle.reshape(x, [x.shape[0], 3 * 32 * 32])

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)