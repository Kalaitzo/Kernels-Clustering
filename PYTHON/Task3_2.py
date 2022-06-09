import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def K(x, h):
    return np.exp(-np.square(np.linalg.norm(x, axis=-1)) / h)


def evaluate(X, X_star, X_circ, a, b, h):
    Rc = [0, 0]
    t = (a * K(X[:, None] - X_star, h)).sum(axis=-1) + (b * K(X[:, None] - X_circ, h)).sum(axis=-1)

    Rc[0] = np.count_nonzero(t>0)
    Rc[1] = np.count_nonzero(t<0)

    return Rc


print('START')
data32 = loadmat('data32.mat')
x_stars, x_circles = data32['stars'], data32['circles']
y_stars, y_circles = np.ones(x_stars.shape[0]), -1*np.ones(x_circles.shape[0])


X = np.append(x_stars, x_circles, axis=0)
Y = np.append(y_stars, y_circles)

for hi in [0.001, 0.01, 0.1]:
    print(f'For h: {hi}')
    for i, li in enumerate([0, 0.1, 1, 10]):
        # print(f'For i:{i}')
        # print(f'For li:{li}')
        # Calculate C
        C = np.linalg.inv(K(X[:, None] - X, hi) + li * np.identity(x_stars.shape[0] + x_circles.shape[0])) @ Y
        a = C[:x_stars.shape[0]]
        b = C[x_circles.shape[0]:]

        total_error = (evaluate(x_stars, x_stars, x_circles, a, b, hi)[1] + evaluate(x_circles, x_stars, x_circles, a, b, hi)[0]) / (x_stars.shape[0] + x_circles.shape[0])
        print(f'Total error is {total_error*100} for lambda: {li}')

        x, y = np.array(np.meshgrid(np.linspace(-1.2, 1.2, 200), np.linspace(-0.1, 1.2, 200)))
        X_ax = np.append(x.reshape(-1,1), y.reshape(-1,1), axis=1)

        t = (a * K(X_ax[:, None] - x_stars, hi)).sum(axis=1) + (b * K(X_ax[:, None] - x_circles, hi)).sum(axis=-1)

        plt.scatter(x_stars[0][0], x_stars[0][1])
        plt.plot(X_ax, t)
        plt.show()
print('DONE')
