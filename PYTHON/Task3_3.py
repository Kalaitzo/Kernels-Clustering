import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


def KMeans(x, K, iterations):
    dist = []
    # Random centers
    z = np.zeros((x.shape[0], K))
    for i in range(K):
        z[:, i] = x[:, np.random.randint(0, x.shape[1])]

    for _ in range(iterations):
        # Initialize C
        C = []
        for i in range(K):
            C.append(np.empty(0, dtype=np.intc))

        # Classify data
        for i, xi in enumerate(x.T):
            class_i = np.argmin(np.linalg.norm(z.T - xi, axis=1))
            C[class_i] = np.append(C[class_i], i)

        # Calculate total variance
        tot_dist = 0
        for i in range(K):
            tot_dist += np.linalg.norm(z[:, i].T - x[:, C[i]].T, axis=1).sum()

        # find new centers
        for i in range(K):
            z[:, i] = x[:, C[i]].mean(axis=1)

        dist.append(tot_dist)
        print(f'Total distance for iteration {_+1} is :{tot_dist}')

    plt.plot(np.array(dist))
    plt.show()

    return z


def classification_error(x, z):
    tot_err_1 = 0
    tot_err_0 = 0

    for i, xi in enumerate(x.T):
        class_i = np.argmin(np.linalg.norm(z.T - xi, axis=1))
        if class_i == 0 and i >= 100:
            tot_err_0 += 1
        elif class_i == 1 and i < 100:
            tot_err_1 += 1

    print(f'Total error for class 0: {tot_err_0} out of 100')
    print(f'Total error for class 1: {tot_err_1} out of 100')
    return (tot_err_0 + tot_err_1) / x.shape[1]


data33 = loadmat('data33.mat')
X = data33['X']

# 3.3 a:
Z = KMeans(X, 2, 30)
print(f'The classification error is: {classification_error(X, Z.T) * 100} %')

# 3.3 b:
X_new = np.append(X, np.square(np.linalg.norm(X, axis=0)).reshape(1, -1), axis=0)
Z = KMeans(X_new, 2, 30)
print(f'The classification error is after we add one more dimension to the points:'
      f' {classification_error(X_new, Z) * 100} %')
