import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Pathway
        self.main_conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.main_conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.main_conv3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.main_pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.main_dropout = nn.Dropout(p=0.5)

        # Branch Pathway
        self.branch_conv = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)

        # Fusion and Classification
        self.global_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        # Main Pathway
        main_path = self.main_conv1(x)
        main_path = self.main_conv2(main_path)
        main_path = self.main_conv3(main_path)
        main_path = self.main_pool(main_path)
        main_path = self.main_dropout(main_path)

        # Branch Pathway
        branch_path = self.branch_conv(x)

        # Fusion
        x = main_path + branch_path

        # Classification
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)