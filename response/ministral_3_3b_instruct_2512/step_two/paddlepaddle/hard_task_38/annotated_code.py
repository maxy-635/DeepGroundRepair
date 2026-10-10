import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the first processing pathway
        self.pathway1 = nn.Sequential(
            nn.LayerNorm([1, 28, 28]),  # Normalize input
            nn.ReLU(),
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),  # Feature extraction
            nn.LayerNorm(normalized_shape='mask'),
            nn.ReLU()
        )

        # Define the second processing pathway
        self.pathway2 = nn.Sequential(
            nn.LayerNorm([1, 28, 28]),  # Normalize input
            nn.ReLU(),
            nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),  # Feature extraction
            nn.LayerNorm([1, 32, 28, 28]),  # Normalize output of conv layer
            nn.ReLU()
        )

        # Define the repeated block structure (executed three times)
        self.repeated_block = nn.Sequential(
            nn.LayerNorm([1, 64, 28, 28]),  # Normalize input
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # Feature extraction
            nn.LayerNorm([1, 64, 28, 28]),  # Normalize output of conv layer
            nn.ReLU()
        )

        # Concatenate the outputs from both pathways
        self.concat = nn.Sequential(
            nn.LayerNorm([1, 64, 28, 28]),
            nn.ReLU()
        )

        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 28 * 28, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Process the input through both pathways
        pathway1_output = self.pathway1(x)
        pathway2_output = self.pathway2(x)

        # Concatenate the outputs from both pathways
        concatenated_output = paddle.concat([pathway1_output, pathway2_output], axis=1)

        # Apply the repeated block structure three times
        for _ in range(3):
            concatenated_output = self.repeated_block(concatenated_output)

        # Classify the output
        output = self.fc_layers(concatenated_output)
        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = paddle.randn((1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)