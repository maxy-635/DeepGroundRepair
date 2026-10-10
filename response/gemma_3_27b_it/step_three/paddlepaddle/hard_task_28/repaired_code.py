import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.LayerNorm(normalized_shape=[32, 32, 32]),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=0)
        )

        # Branch Path
        self.branch_path = nn.Identity()  # Direct connection to input

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32 * 32 * 32, out_features=128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        main_output = self.main_path(x)  # output tensor shape of this layer should be (1, 3, 32, 32)
        branch_output = self.branch_path(x)  # output tensor shape: (1, 3, 32, 32)
        combined_output = main_output + branch_output
        flattened_output = self.flatten(combined_output)
        fc1_output = self.fc1(flattened_output)
        relu_output = self.relu(fc1_output)
        output = self.fc2(relu_output)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Example input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)