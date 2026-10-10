class DL_Model:
    def __init__(self):
        self.pool1 = paddle.nn.MaxPool2D(kernel_size=(2, 2), stride=(2, 2))
        self.flatten1 = paddle.nn.Flatten()
        self.pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.flatten2 = paddle.nn.Flatten()
        self.pool3 = paddle.nn.MaxPool2D(kernel_size=4, stride=4)
        self.flatten3 = paddle.nn.Flatten()
        self.concat = paddle.concat
        self.linear1 = paddle.nn.Linear(1280, 512)
        self.linear2 = paddle.nn.Linear(512, 10)

    def __call__(self, x):
        x1 = self.pool1(x)
        x1 = self.flatten1(x1)
        x2 = self.pool2(x)
        x2 = self.flatten2(x2)
        x3 = self.pool3(x)
        x3 = self.flatten3(x3)
        x = self.concat([x1, x2, x3], axis=1)
        x = self.linear1(x)
        x = paddle.nn.functional.relu(x)
        x = self.linear2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.zeros([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)