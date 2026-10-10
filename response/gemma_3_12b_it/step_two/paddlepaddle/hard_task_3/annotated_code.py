import paddle
import paddle.nn as nn
import paddle.static as static

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Pathway
        self.conv1_main_1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv1_main_2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout_main = nn.Dropout2D(p=0.25)
        self.conv1_main_3 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv1_main_4 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.dropout_main_2 = nn.Dropout2D(p=0.25)


        # Branch Pathway
        self.conv1_branch = nn.Conv2D(in_channels=3, out_channels=256, kernel_size=1, stride=1, padding=0)

        # Fully Connected Layer
        self.fc = nn.Linear(in_features='mask', out_features=10)


    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main Pathway
        x1 = self.conv1_main_1(x)  # output tensor shape: (1, 32, 32, 32)
        x1 = self.conv1_main_2(x1)  # output tensor shape: (1, 64, 32, 32)
        x1 = self.dropout_main(x1)  # output tensor shape: (1, 64, 32, 32)
        x1 = self.conv1_main_3(x1)  # output tensor shape: (1, 128, 32, 32)
        x1 = self.conv1_main_4(x1)  # output tensor shape: (1, 256, 32, 32)
        x1 = self.dropout_main_2(x1)  # output tensor shape: (1, 256, 32, 32)

        # Branch Pathway
        x2 = self.conv1_branch(x)  # output tensor shape: (1, 256, 32, 32)

        # Combine Pathways
        combined = paddle.add(x1, x2)  # output tensor shape: (1, 256, 32, 32)

        # Reshape for Fully Connected Layer
        combined = paddle.reshape(combined, [combined.shape[0], -1])  # output tensor shape: (1, 262144)

        # Fully Connected Layer
        output = self.fc(combined)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)