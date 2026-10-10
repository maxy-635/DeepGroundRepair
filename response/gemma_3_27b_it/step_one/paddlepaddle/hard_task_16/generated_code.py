import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1_group1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2_group1 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3_group1 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)

        self.conv1_group2 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2_group2 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3_group2 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)

        self.conv1_group3 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2_group3 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3_group3 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)

        # Transition Convolution
        self.transition_conv = nn.Conv2D(in_channels=48, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Block 2
        self.global_max_pool = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=32, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        
        # Branch
        self.branch_conv = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Classification
        self.fc_classifier = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        # Block 1
        group1 = F.relu(self.conv3_group1(F.relu(self.conv2_group1(F.relu(self.conv1_group1(x))))))
        group2 = F.relu(self.conv3_group2(F.relu(self.conv2_group2(F.relu(self.conv1_group2(x))))))
        group3 = F.relu(self.conv3_group3(F.relu(self.conv2_group3(F.relu(self.conv1_group3(x))))))
        block1_output = paddle.concat([group1, group2, group3], axis=1)

        # Transition Convolution
        adjusted_output = F.relu(self.transition_conv(block1_output))

        # Block 2
        global_pooled = self.global_max_pool(adjusted_output)
        flattened = paddle.nn.Flatten()(global_pooled)
        weights = F.relu(self.fc2(F.relu(self.fc1(flattened))))
        
        weights = paddle.reshape(weights, shape=[weights.shape[0], 32, 1, 1])
        main_path_output = adjusted_output * weights
        
        # Branch
        branch_output = self.branch_conv(x)
        
        # Add main path and branch
        combined_output = main_path_output + branch_output

        # Classification
        classification_output = self.fc_classifier(paddle.nn.Flatten()(combined_output))
        return classification_output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)