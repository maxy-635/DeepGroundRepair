import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: dual-path structure
        self.main_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.branch_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        )
        self.addition = nn.Add()

        # Second block: depthwise separable convolutional layers
        self.depthwise_conv_1x1 = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.depthwise_conv_3x3 = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.depthwise_conv_5x5 = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.concat = nn.Concat(axis=1)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32*32*3, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=10)

    def forward(self, x):
        # First block: dual-path structure
        main_path_output = self.main_path(x)
        branch_path_output = self.branch_path(x)
        combined_output = self.addition(main_path_output, branch_path_output)

        # Second block: depthwise separable convolutional layers
        conv_1x1_output = self.depthwise_conv_1x1(combined_output)
        conv_3x3_output = self.depthwise_conv_3x3(combined_output)
        conv_5x5_output = self.depthwise_conv_5x5(combined_output)

        # Concatenate outputs
        concatenated_output = self.concat([conv_1x1_output, conv_3x3_output, conv_5x5_output])

        # Fully connected layers
        flattened_output = self.flatten(concatenated_output)
        fc_output = self.fc1(flattened_output)
        fc_output = paddle.nn.functional.relu(fc_output)
        fc_output = self.fc2(fc_output)

        return fc_output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.zeros([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)