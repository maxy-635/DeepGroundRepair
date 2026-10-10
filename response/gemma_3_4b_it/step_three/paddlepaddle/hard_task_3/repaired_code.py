import paddle
import paddle.nn as nn
import paddle.io
import paddle.vision.datasets as datasets

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the main pathway
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.25)

        self.conv3 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.conv4 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(p=0.25)

        self.conv5 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.relu3 = nn.ReLU()
        self.conv6 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.dropout3 = nn.Dropout(p=0.25)

        # Concatenate the outputs from the three groups
        self.concat = nn.Concat(dim=0)

        # Define the branch pathway
        self.conv7 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.relu4 = nn.ReLU()

        # Combine the main and branch pathways
        self.add = nn.Add()

        # Define the fully connected layer
        self.fc = nn.fc(in_features=32 * 6,
                        size=10,
                        activation=nn.ReLU(),
                        name='fc')

    def forward(self, x):
        # Split the input into three groups along the channel dimension
        x1 = x
        x2 = x
        x3 = x

        # Process each group
        x1 = self.conv1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.dropout1(x1)

        x2 = self.conv3(x2)
        x2 = self.relu2(x2)
        x2 = self.conv4(x2)
        x2 = self.dropout2(x2)

        x3 = self.conv5(x3)
        x3 = self.relu3(x3)
        x3 = self.conv6(x3)
        x3 = self.dropout3(x3)

        # Concatenate the outputs from the three groups
        x = self.concat([x1, x2, x3])

        # Process the branch pathway
        x = self.conv7(x)
        x = self.relu4(x)

        # Combine the main and branch pathways
        x = self.add([x, x])

        # Pass the resulting output through a fully connected layer
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor
    input_tensor = paddle.zeros((1, 3, 32, 32), dtype=paddle.float32)
    # Pass the input tensor through the model
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)