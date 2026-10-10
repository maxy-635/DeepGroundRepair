import paddle
import paddle.nn as nn
import paddle.io
import paddle.vision.datasets as datasets

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        self.main_path = nn.Sequential(
            nn.AdaptiveAvgPool2D((1, 1)),
            nn.FullyConnected(in_features=32*32*3, out_features=64),
            nn.FullyConnected(in_features=64, out_features=10)
        )

        self.branch_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.FullyConnected(in_features=32*32*10, out_features=64)
        )

        self.fusion_path = nn.FullyConnected(in_features=64, out_features=10)

        self.fc1 = nn.FullyConnected(in_features=10, out_features=10)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        main_output = self.main_path(x)
        branch_output = self.branch_path(x)

        main_output_reshaped = main_output.reshape(x.shape[0], -1)
        weight = self.main_path(x)
        weight = weight.reshape(x.shape[0], -1)
        weight = paddle.full((x.shape[0], x.shape[1]), 1.0)
        weight = paddle.scatter_nd_add(weight, paddle.ones_like(weight), main_output_reshaped)
        weight = paddle.mean(weight, axis=1, keepdim=True)
        weight = weight.repeat(1, x.shape[1])
        
        combined_output = main_output + branch_output
        final_output = self.fusion_path(combined_output)
        output = self.fc1(final_output)
        output = self.softmax(output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(batch_size=64, channels=3, height=32, width=32, dtype='float32')
    output_tensor = model(input_tensor)
    print(output_tensor.shape)