import paddle
import paddle.nn as nn
import paddle.vision.datasets as datasets
import paddle.io

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Branch path
        self.conv_branch = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Combine paths
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=128 * 8 * 8, out_features=10)  # Adjusted input size

    def forward(self, x):
        # Main path
        main_path = x
        main_path = self.conv1(main_path)
        main_path = self.dropout1(main_path)
        main_path = self.conv2(main_path)
        main_path = self.dropout2(main_path)
        main_path = self.conv3(main_path)

        # Branch path
        branch_path = x
        branch_path = self.conv_branch(branch_path)

        # Combine paths
        combined = main_path + branch_path
        combined = self.flatten(combined)
        output = self.fc(combined)
        output = nn.Softmax(axis=-1)(output)

        return output


if __name__ == "__main__":
    # Instantiate the model
    model = DL_Model()

    # Create a dummy input tensor
    input_tensor = paddle.randn((1, 1, 28, 28))  # Batch size 1, 1 channel, 28x28 image

    # Pass the input through the model
    output_tensor = model(input_tensor)

    # Print the output shape
    print("Output shape:", output_tensor.shape)