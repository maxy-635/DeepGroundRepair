import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: main path and branch path
        self.main_path_conv1 = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.main_path_dropout1 = nn.Dropout(p=0.5)
        self.main_path_conv2 = nn.Conv2D(
            in_channels=64,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Second block: separable convolutions with different kernel sizes
        self.sep_conv1 = nn.Conv2D(
            in_channels=3,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.sep_conv2 = nn.Conv2D(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.sep_conv3 = nn.Conv2D(
            in_channels=3,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.dropout_sep = nn.Dropout(p=0.5)

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=32 * 8 * 8, out_features=10)

    def forward(self, x):
        # First block
        main_path = self.main_path_conv1(x)
        main_path = self.main_path_dropout1(main_path)
        main_path = self.main_path_conv2(main_path)
        output_block1 = paddle.add(x, main_path)

        # Second block
        x_split = paddle.split(output_block1, num_or_sections=3, axis=1)
        conv1 = self.sep_conv1(x_split[0])
        conv2 = self.sep_conv2(x_split[1])
        conv3 = self.sep_conv3(x_split[2])

        conv1 = self.dropout_sep(conv1)
        conv2 = self.dropout_sep(conv2)
        conv3 = self.dropout_sep(conv3)

        output_block2 = paddle.concat([conv1, conv2, conv3], axis=1)

        # Final layers
        x = self.flatten(output_block2)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected shape: [1, 10]