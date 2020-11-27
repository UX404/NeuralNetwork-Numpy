from Layer import *


class Linear_2():
    def __init__(self, n):
        self.hidden_layer = Linear(784, n)
        self.output_layer = Linear(n, 10)
        self.loss = MSE()

    def forward(self, x):
        f1 = self.hidden_layer.forward(x)
        f2 = self.output_layer.forward(f1)
        f3 = self.loss.forward(f2)
        return f3

    def backward(self, truth, learning_rate):
        b1 = self.loss.backward(truth)
        b2 = self.output_layer.backward(b1, learning_rate)
        b3 = self.hidden_layer.backward(b2, learning_rate)
        return

    def evaluate_loss(self, y, truth):  # Loss函数：MSE
        loss = np.average(0.5 * (y - truth) ** 2)
        return loss
