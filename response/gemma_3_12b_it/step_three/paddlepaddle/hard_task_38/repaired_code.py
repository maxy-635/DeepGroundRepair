import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.pathway1 = nn.Sequential(
            nn.BatchNorm2D(num_features=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
        )
        self.pathway2 = nn.Sequential(
            nn.BatchNorm2D(num_features=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
        )

        # Repeat the block structure three times
        self.repeated_block1_pathway1 = self.pathway1
        self.repeated_block2_pathway1 = self.pathway1
        self.repeated_block3_pathway1 = self.pathway1

        self.repeated_block1_pathway2 = self.pathway2
        self.repeated_block2_pathway2 = self.pathway2
        self.repeated_block3_pathway2 = self.pathway2


        self.merge_layer = nn.Conv2D(in_channels=32 * 2, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Pathway 1
        x1 = self.repeated_block1_pathway1(x)
        x1 = paddle.concat([x, x1], axis=1)
        x1 = self.repeated_block2_pathway1(x1)
        x1 = paddle.concat([x, x1], axis=1)
        x1 = self.repeated_block3_pathway1(x1)

        # Pathway 2
        x2 = self.repeated_block1_pathway2(x)
        x2 = paddle.concat([x, x2], axis=1)
        x2 = self.repeated_block2_pathway2(x2)
        x2 = paddle.concat([x, x2], axis=1)
        x2 = self.repeated_block3_pathway2(x2)

        # Merge pathways
        merged = paddle.concat([x1, x2], axis=1)
        merged = self.merge_layer(merged)
        merged = merged.flatten()
        
        # Fully connected layers
        x = self.fc1(merged)
        x = paddle.nn.ReLU()(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)