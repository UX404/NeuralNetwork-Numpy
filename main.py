from Net import *
from time import time


def one_hot_labels(labels):
    labels_onehot = np.zeros((labels.size, 10))
    for n in range(labels.size):
        labels_onehot[n][int(labels[n][0])] = 1
    return labels_onehot


x_train = np.load('data/train_images.npy') / 255
x_test = np.load('data/test_images.npy') / 255
y_train = one_hot_labels(np.load('data/train_labels.npy'))
y_test = one_hot_labels(np.load('data/test_labels.npy'))

n = 100
epochs = 50
LEARNING_RATE = 0.1

net = Linear_2(n)

clock = time()
for epoch in range(epochs):
    # train
    print('Training Epoch {epoch}...'.format(epoch=epoch+1))
    for n in range(60000):
        # print('Pic number:{n}'.format(n=n))
        y = net.forward(x_train[n:n+1])
        net.backward(y_train[n:n+1], LEARNING_RATE)

    # test
    print('Evaluating...')
    predicted_correctly = []
    losses = []
    for n in range(10000):
        y = net.forward(x_test[n:n+1])
        predicted_correctly.append(np.argmax(y[0]) == np.argmax(y_test[n]))
        if epoch == epochs-1:
            print('Predicted: {predicted}, Truth: {truth}'.format(predicted=np.argmax(y[0]), truth=np.argmax(y_test[n])))
        losses.append(net.evaluate_loss(y, y_test[n:n+1]))
    print('Epoch: {epoch}, Loss: {loss:.5f}, Accuracy: {acc:.2f}%'.format(epoch=epoch+1, loss=sum(losses)/len(losses), acc=sum(predicted_correctly)/len(predicted_correctly)*100))

print('Average time per epoch: {time:.3f}s'.format(time=(time()-clock)/epochs))
