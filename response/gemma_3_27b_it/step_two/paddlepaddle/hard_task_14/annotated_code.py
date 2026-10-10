import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distribution import ReshapeTransform

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.avg_pool = nn.AvgPool2D(kernel_size=8, stride=8, padding=0)
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=3)
        self.reshape = ReshapeTransform(in_event_shape='mask', out_event_shape='mask')

        # Branch Path
        self.conv = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

        # Combined Layers
        self.fc3 = nn.Linear(in_features=3 * 32 * 32, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=64)
        self.fc5 = nn.Linear(in_features=64, out_features=10)
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        # Main Path
        main_path = self.avg_pool(x)
        main_path = main_path.reshape([main_path.shape[0], -1])
        main_path = self.fc1(main_path)
        main_path = self.fc2(main_path)
        main_path = self.reshape(main_path)
        main_path = main_path.transpose([0, 2, 3, 1])
        
        # Branch Path
        branch_path = self.conv(x)
        branch_path = self.relu(branch_path)

        # Combine Paths
        combined = paddle.add(x, branch_path)

        # Combined Layers
        combined = combined.reshape([combined.shape[0], -1])
        combined = self.fc3(combined)
        combined = self.fc4(combined)
        combined = self.fc5(combined)
        output = self.softmax(combined)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)