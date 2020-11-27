import numpy as np


class Linear():
    def __init__(self, input_num, output_num):
        self.weights = (np.random.rand(input_num, output_num) - 0.5) / 10  # [-0.05, 0.05]
        self.bias = np.ones((1, output_num))  # 1
        self.x = None
        self.m = None
        self.y = None

    def forward(self, x):
        self.x = x
        self.m = np.dot(self.x, self.weights) + self.bias  # 线性层
        self.y = 1 / (1 + np.exp(-self.m))  # 激活函数 Sigmoid
        return self.y

    def backward(self, dy, learning_rate):
        dm = dy * self.y * (1 - self.y)  # 激活函数BP
        dw = np.dot(self.x.T, dm)  # 线性层BP
        db = dm
        dx = np.dot(dm, self.weights.T)
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db
        return dx


class MSE():
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return self.x

    def backward(self, truth):
        return (self.x - truth) / self.x.size
