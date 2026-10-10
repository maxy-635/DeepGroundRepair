import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=1, stride=1, padding=0)
        self.pool1_1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.pool1_2 = nn.AvgPool2d(kernel_size=3, stride=3, padding=1)
        self.pool1_3 = nn.AvgPool2d(kernel_size=4, stride=4, padding=2)

        self.flatten1_1 = nn.Flatten()
        self.dropout1_1 = nn.Dropout(p=0.25)

        self.flatten1_2 = nn.Flatten()
        self.dropout1_2 = nn.Dropout(p=0.25)

        self.flatten1_3 = nn.Flatten()
        self.dropout1_3 = nn.Dropout(p=0.25)

        self.concat1 = nn.Concat('mask')

        self.fc1 = nn.Linear(3 * 32, 128)
        self.reshape1 = nn.Linear(128, 4)


        # Block 2
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)
        self.conv2_3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.pool2_1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2_1_out = nn.Flatten()

        self.pool2_2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2_2_out = nn.Flatten()

        self.pool2_3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2_3_out = nn.Flatten()


        self.concat2 = nn.Concat([self.conv2_1_out, self.conv2_2_out, self.conv2_3_out])

        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)  # Output layer

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.pool1_1(x)
        x = self.pool1_2(x)
        x = self.pool1_3(x)

        x = self.flatten1_1(x)
        x = self.dropout1_1(x)

        x = self.flatten1_2(x)
        x = self.dropout1_2(x)

        x = self.flatten1_3(x)
        x = self.dropout1_3(x)

        x = self.concat1

        x = self.fc1(x)
        x = self.reshape1(x)

        # Block 2
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = self.pool2_1(x)
        x = self.conv2_1_out(x)

        x = self.pool2_2(x)
        x = self.conv2_2_out(x)

        x = self.pool2_3(x)
        x = self.conv2_3_out(x)

        x = self.concat2

        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)