from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

X, Y = datasets.make_moons(n_samples= 100, noise=0.1)
color = ['blue' if k == 0 else 'red' for k in Y]

plt.scatter(X[:,0], X[:, 1], c = color)
# plt.savefig('duas_luas.svg')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def run_neural_net(x0, x1, w00, w01, b0, w10, w11, b1, w20, w21, b2):
    s00 = x0 * w00
    s01 = x1 * w01
    v0 = b0 + s00 + s01
    y0 = sigmoid(v0)

    s10 = x0 * w10
    s11 = x1 * w11
    v1 = b1 + s10 + s11
    y1 = sigmoid(v1)

    s20 = y0 * w20
    s21 = y1 * w21
    v2 = b2 + s20 + s21
    y2 = sigmoid(v2)
    return 1 if y2 > 0.5 else 0


def neural_net(x0, x1, d, w00, w01, b0, w10, w11, b1, w20, w21, b2):
    s00 = x0 * w00
    s01 = x1 * w01
    v0 = b0 + s00 + s01
    y0 = sigmoid(v0)

    s10 = x0 * w10
    s11 = x1 * w11
    v1 = b1 + s10 + s11
    y1 = sigmoid(v1)

    s20 = y0 * w20
    s21 = y1 * w21
    v2 = b2 + s20 + s21
    y2 = sigmoid(v2)

    e = y2 - d
    L = e ** 2

    grad_L = 1
    grad_e = 2 * e * grad_L
    grad_y2 = grad_e
    grad_v2 = grad_y2 * (y2 * (1 - y2))
    grad_b2 = grad_v2
    grad_s21 = grad_v2
    grad_s20 = grad_v2
    grad_w21 = grad_s21 * y1
    grad_w20 = grad_s20 * y0

    grad_y1 = grad_s21 * w21
    grad_y0 = grad_s20 * w20

    grad_v0 = grad_y0 * (y0 * (1 - y0))
    grad_v1 = grad_y1 * (y1 * (1 - y1))

    grad_s00 = grad_v0
    grad_s01 = grad_v0

    grad_s10 = grad_v1
    grad_s11 = grad_v1

    grad_w00 = grad_s00 * x0
    grad_w01 = grad_s01 * x1
    grad_b0 = grad_v0

    grad_w10 = grad_s10 * x0
    grad_w11 = grad_s11 * x1
    grad_b1 = grad_v1

    return grad_w00, grad_w01, grad_w10, grad_w11, grad_w20, grad_w21, grad_b0, grad_b1, grad_b2, L

w00 = w01 = b0 = w10 = w11 = b1 = w20 = w21 = b2 = 0
taxa = 0.1

acc = 0
for i in range(100):
    out = run_neural_net(X[i, 0], X[i, 1], w00, w01, b0, w10, w11, b1, w20, w21, b2)
    if out == Y[i]:
        acc += 1
print(acc)

for i in range(10000):
    loss = 0
    grad_w00 = np.random.rand() - 0.5
    grad_w01 = np.random.rand() - 0.5
    grad_b0 = np.random.rand() - 0.5
    grad_w10 = np.random.rand() - 0.5
    grad_w11 = np.random.rand() - 0.5
    grad_b1 = np.random.rand() - 0.5
    grad_w20 = np.random.rand() - 0.5
    grad_w21 = np.random.rand() - 0.5
    grad_b2 = np.random.rand() - 0.5
    for k in range(100):
        gw00, gw01, gb0, gw10, gw11, gb1, gw20, gw21, gb2, L = neural_net(X[k, 0], X[k, 1], Y[k], w00, w01, b0, w10,
                                                                          w11, b1, w20, w21, b2)

        grad_w00 += gw00
        grad_w01 += gw01
        grad_b0 += gb0
        grad_w10 += gw10
        grad_w11 += gw11
        grad_b1 += gb1
        grad_w20 += gw20
        grad_w21 += gw21
        grad_b2 += gb2
        loss += L

    w00 -= taxa * grad_w00
    w01 -= taxa * grad_w01
    b0 -= taxa * grad_b0

    w10 -= taxa * grad_w10
    w11 -= taxa * grad_w11
    b1 -= taxa * grad_b1

    w20 -= taxa * grad_w20
    w21 -= taxa * grad_w21
    b2 -= taxa * grad_b2

acc = 0
for i in range(100):
    out = run_neural_net(X[i, 0], X[i, 1], w00, w01, b0, w10, w11, b1, w20, w21, b2)
    if out == Y[i]:
        acc += 1
print(acc)

model = Sequential()
model.add(Dense(units=20, input_dim=2, activation='relu'))
model.add(Dense(units=10, input_dim=2, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
opt = SGD(learning_rate=taxa)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

model.fit(X, Y, epochs=100, verbose=False, batch_size=5)

acc = model.evaluate(X, Y)