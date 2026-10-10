import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Pathway 1
        pathway1 = nn.Sequential()
        for _ in range(3):
            pathway1.add_sublayer(nn.BatchNorm2D(num_features=1), sublayer_name='bn1')
            pathway1.add_sublayer(nn.ReLU(), sublayer_name='relu1')
            pathway1.add_sublayer(nn.Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1), sublayer_name='conv1')

        # Pathway 2
        pathway2 = nn.Sequential()
        for _ in range(3):
            pathway2.add_sublayer(nn.BatchNorm2D(num_features=1), sublayer_name='bn2')
            pathway2.add_sublayer(nn.ReLU(), sublayer_name='relu2')
            pathway2.add_sublayer(nn.Conv2D(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1), sublayer_name='conv2')

        self.pathway1 = pathway1
        self.pathway2 = pathway2

        # Classification layers
        self.fc1 = nn.Linear(in_features=784, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Process through pathways
        x1 = self.pathway1(x)
        x2 = self.pathway2(x)

        # Concatenate outputs
        x_concat = paddle.concat([x, x1, x2], axis=1)

        # Flatten
        x_flat = paddle.reshape(x_concat, [x_concat.shape[0], -1])

        # Fully connected layers
        x_fc1 = F.relu(self.fc1(x_flat))
        x_fc2 = self.fc2(x_fc1)

        return x_fc2


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape) # Expected output shape: [1, 10]