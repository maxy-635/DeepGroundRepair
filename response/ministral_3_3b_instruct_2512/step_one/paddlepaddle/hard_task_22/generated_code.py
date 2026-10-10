import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_model, self).__init__()
        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2D(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2D(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2D(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Concatenate the outputs of main and branch paths
        self.concatenate = nn.Sequential(
            nn.Conv2D(128, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        main_output = self.main_path(x)
        branch_output = self.branch_path(x)
        # Concatenate the outputs along the channel dimension
        concatenated = paddle.concat([main_output, branch_output], axis=1)
        fused_output = self.concatenate(concatenated)
        output = self.fc(fused_output)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Example input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)