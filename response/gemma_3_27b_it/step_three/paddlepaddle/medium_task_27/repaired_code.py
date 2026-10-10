import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 3x3 convolutional kernel
        self.conv3x3_1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1_1 = nn.ReLU()
        self.conv3x3_2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1_2 = nn.ReLU()

        # Branch 2: 5x5 convolutional kernel
        self.conv5x5_1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2_1 = nn.ReLU()
        self.conv5x5_2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.relu2_2 = nn.ReLU()

        # Combine features
        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.flatten = nn.Flatten()

        # Attention weights
        self.fc_attention_1 = nn.Linear(in_features=64, out_features=32)
        self.relu_attention_1 = nn.ReLU()
        self.softmax_attention_1 = nn.Softmax(axis=1)

        self.fc_attention_2 = nn.Linear(in_features=64, out_features=32)
        self.relu_attention_2 = nn.ReLU()
        self.softmax_attention_2 = nn.Softmax(axis=1)
        
        # Final classification layer
        self.fc_final = nn.Linear(in_features=64, out_features=10)


    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Branch 1
        branch1 = self.conv3x3_1(x)  # output tensor shape: (1, 32, 32, 32)
        branch1 = self.relu1_1(branch1)  # output tensor shape: (1, 32, 32, 32)
        branch1 = self.conv3x3_2(branch1)  # output tensor shape: (1, 64, 32, 32)
        branch1 = self.relu1_2(branch1)  # output tensor shape: (1, 64, 32, 32)

        # Branch 2
        branch2 = self.conv5x5_1(x)  # output tensor shape: (1, 32, 32, 32)
        branch2 = self.relu2_1(branch2)  # output tensor shape: (1, 32, 32, 32)
        branch2 = self.conv5x5_2(branch2)  # output tensor shape: (1, 64, 32, 32)
        branch2 = self.relu2_2(branch2)  # output tensor shape: (1, 64, 32, 32)

        # Combine features
        combined = branch1 + branch2  # output tensor shape: (1, 64, 32, 32)

        # Global Average Pooling and Flatten
        pooled = self.avg_pool(combined)  # output tensor shape: (1, 64, 1, 1)
        flattened = self.flatten(pooled)  # output tensor shape: (1, 64)

        #Attention weights
        attention_weight_1 = self.fc_attention_1(flattened)  # output tensor shape: (1, 32)
        attention_weight_1 = self.relu_attention_1(attention_weight_1)  # output tensor shape: (1, 32)
        attention_weight_1 = self.softmax_attention_1(attention_weight_1)  # output tensor shape: (1, 32)

        attention_weight_2 = self.fc_attention_2(flattened)  # output tensor shape: (1, 32)
        attention_weight_2 = self.relu_attention_2(attention_weight_2)  # output tensor shape: (1, 32)
        attention_weight_2 = self.softmax_attention_2(attention_weight_2)  # output tensor shape: (1, 32)
        
        # Weighted output
        weighted_output = attention_weight_1 * branch1 + attention_weight_2 * branch2  # output tensor shape: (1, 64, 32, 32)

        # Final Classification
        final_output = self.fc_final(flattened)
        

        return final_output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)