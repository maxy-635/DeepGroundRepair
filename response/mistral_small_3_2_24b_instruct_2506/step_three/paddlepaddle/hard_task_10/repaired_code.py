import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()

        # First feature extraction path: 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Second feature extraction path: 1x1, 1x7, 7x1 convolutions
        self.path2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )

        # 1x1 convolution to align dimensions
        self.concat_conv = nn.Conv2D(in_channels=128, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Branch connection
        self.branch = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        )

        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(in_features=32*32*3, out_features=512),
            nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, x):
        # Feature extraction paths
        path1_out = self.path1(x)
        path2_out = self.path2(x)

        # Concatenate and apply 1x1 convolution
        concat_out = paddle.concat([path1_out, path2_out], axis=1)
        main_out = self.concat_conv(concat_out)

        # Branch connection
        branch_out = self.branch(x)

        # Merge outputs
        merged_out = paddle.add(main_out, branch_out)

        # Flatten and classify
        flattened = paddle.flatten(merged_out, start_axis=1)
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # CIFAR-10 image shape: [batch_size, 3, 32, 32]
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]