import paddle
import paddle.nn as nn

class DL_Model(paddle.nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch1_conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch2_conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch3_conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.avgpool2d1 = nn.AvgPool2D(kernel_size=2, stride=2)
        self.transposeconv2d1 = nn.Conv2DTranspose(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.transposeconv2d2 = nn.Conv2DTranspose(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)

        self.concat_dim = 1
        self.main_conv2 = nn.Conv2D(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)

        self.fc = paddle.static.nn.fc(size=10, activation='softmax')

    def forward(self, x):
        # Main path
        x = self.main_conv1(x)

        # Branch paths
        branch1 = self.branch1_conv1(x)
        branch2 = self.avgpool2d1(x)
        branch2 = self.transposeconv2d1(branch2)
        branch3 = self.avgpool2d1(x)
        branch3 = self.transposeconv2d2(branch3)

        # Concatenate
        main_path_output = self.main_conv2(x)
        branch_output = paddle.concat([branch1, branch2, branch3], axis=self.concat_dim)
        
        # Fuse
        output = paddle.add(main_path_output, branch_output)
        
        # Fully connected layer
        output = self.fc(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)