import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.avg_pool = nn.AvgPool2D(kernel_size=input_tensor.shape[2], stride=1, padding=0)
        self.fc1 = nn.Linear(in_features='mask', out_features=input_tensor.shape[1])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=input_tensor.shape[1], out_features=input_tensor.shape[1])

        # Branch path
        self.conv = nn.Conv2D(in_channels=input_tensor.shape[1], out_channels=input_tensor.shape[1], kernel_size=3, stride=1, padding=1)

        # Fully connected layers for final classification
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(in_features=input_tensor.shape[1] * input_tensor.shape[2] * input_tensor.shape[3], out_features=10)
        self.relu2 = nn.ReLU()
        self.fc4 = nn.Linear(in_features=10, out_features=10)
        self.relu3 = nn.ReLU()
        self.fc5 = nn.Linear(in_features=10, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (32, 3, 32, 32)
        # Main path
        pooled_output = self.avg_pool(x)  # output tensor shape: (32, 3, 1, 1)
        flattened_output = self.flatten(pooled_output)  # output tensor shape: (32, 3)
        fc1_output = self.fc1(flattened_output)
        relu1_output = self.relu1(fc1_output)
        fc2_output = self.fc2(relu1_output)
        fc2_output = fc2_output.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        # Branch path
        conv_output = self.conv(x)

        # Combine paths
        combined_output = fc2_output + conv_output

        # Final classification
        flattened_combined = self.flatten(combined_output)
        fc3_output = self.fc3(flattened_combined)
        relu2_output = self.relu2(fc3_output)
        fc4_output = self.fc4(relu2_output)
        relu3_output = self.relu3(fc4_output)
        output = self.fc5(relu3_output)

        return output


if __name__ == "__main__":
    # Dummy input tensor for CIFAR-10 (batch_size, channels, height, width)
    input_tensor = paddle.randn(shape=[32, 3, 32, 32])  # Example batch size of 32
    model = DL_Model()
    output_tensor = model(input_tensor)
    print(output_tensor.shape)