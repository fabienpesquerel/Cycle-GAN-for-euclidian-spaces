#!/usr/bin/python3

import numpy as np
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

# Write a csv


def write_csv(path, data):
    np.savetxt(path, data, delimiter=",")


# Generate spirals

def two_spirals(N, noise=2.1):
    n = np.sqrt(np.random.rand(N, 1)) * 780 * (2*np.pi)/360
    dx = - np.cos(n) * n + np.random.rand(N, 1) * noise
    dy = np.sin(n) * n + np.random.rand(N, 1) * noise
    return (np.vstack((np.hstack((dx, dy)), np.hstack((-dx, -dy)))),
            np.hstack((np.zeros(N), np.ones(N))))


# Generate gaussian2gaussian dataset
# A
NA = 1000
covA = [[1., 0.], [0., 1.]]
meanA = [0.7, 0.1]
A_train = np.random.multivariate_normal(meanA, covA, NA)
A_test = np.random.multivariate_normal(meanA, covA, NA)
write_csv('datasets/gaussian2gaussian/train/A/gaussian.csv', A_train)
write_csv('datasets/gaussian2gaussian/test/A/gaussian.csv', A_test)

# B
NB = 1000
covB = [[3., 0.], [0., 0.1]]
meanB = [-3., 2.]
B_train = np.random.multivariate_normal(meanB, covB, NB)
B_test = np.random.multivariate_normal(meanB, covB, NB)
write_csv('datasets/gaussian2gaussian/train/B/gaussian.csv', B_train)
write_csv('datasets/gaussian2gaussian/test/B/gaussian.csv', B_test)


# Generate halfMoon2gaussian dataset
N = 4000
X, Y = make_moons(2*N, shuffle=True, noise=.1)

# A
A_train = X[Y == 0][:N//2]
A_test = X[Y == 0][N//2:]
write_csv('datasets/halfMoon2gaussian/train/A/halfMoon.csv', A_train)
write_csv('datasets/halfMoon2gaussian/test/A/halfMoon.csv', A_test)
# plt.scatter(A_train[:, 0], A_train[:, 1])
# plt.show()

# B
NB = 1000
covB = [[3., 0.], [0., 0.1]]
meanB = [-3., 2.]
B_train = np.random.multivariate_normal(meanB, covB, NB)
B_test = np.random.multivariate_normal(meanB, covB, NB)
write_csv('datasets/halfMoon2gaussian/train/B/gaussian.csv', B_train)
write_csv('datasets/halfMoon2gaussian/test/B/gaussian.csv', B_test)


# Generate circle2gaussian
N = 4000
X, Y = make_circles(2*N, shuffle=True, noise=.1)

# A
A_train = X[Y == 0][:N//2]
A_test = X[Y == 0][N//2:]
write_csv('datasets/circle2gaussian/train/A/circle.csv', A_train)
write_csv('datasets/circle2gaussian/test/A/circle.csv', A_test)
print(len(A_test))
# plt.scatter(A_train[:, 0], A_train[:, 1])
# plt.show()

# B
NB = 1000
covB = [[1., 0.], [0., 0.1]]
meanB = [-1., 2.]
B_train = np.random.multivariate_normal(meanB, covB, NB)
B_test = np.random.multivariate_normal(meanB, covB, NB)
write_csv('datasets/circle2gaussian/train/B/gaussian.csv', B_train)
write_csv('datasets/circle2gaussian/test/B/gaussian.csv', B_test)


# Generate gaussian2spiral

# A
NA = 1000
covA = [[1., 0.], [0., 1.]]
meanA = [0.4, 0.1]
A_train = np.random.multivariate_normal(meanA, covA, NA)
A_test = np.random.multivariate_normal(meanA, covA, NA)
write_csv('datasets/gaussian2spiral/train/A/gaussian.csv', A_train)
write_csv('datasets/gaussian2spiral/test/A/gaussian.csv', A_test)

# B
N = 5000
X, Y = two_spirals(2*N, noise=3.)
B_train = X[Y == 0][:N//2]
B_test = X[Y == 0][N//2:]
write_csv('datasets/gaussian2spiral/train/B/spiral.csv', B_train)
write_csv('datasets/gaussian2spiral/test/B/spiral.csv', B_test)
# plt.scatter(B_train[:, 0], B_train[:, 1])
# plt.show()


# Generate halfMoon2spiral
N = 4000
X, Y = make_moons(2*N, shuffle=True, noise=.1)

# A
A_train = X[Y == 0][:N//2]
A_test = X[Y == 0][N//2:]
write_csv('datasets/halfMoon2spiral/train/A/halfMoon.csv', A_train)
write_csv('datasets/halfMoon2spiral/test/A/halfMoon.csv', A_test)

# B
N = 4000
X, Y = two_spirals(2*N)
B_train = X[Y == 0][:N//2]
B_test = X[Y == 0][N//2:]
write_csv('datasets/halfMoon2spiral/train/B/spiral.csv', B_train)
write_csv('datasets/halfMoon2spiral/test/B/spiral.csv', B_test)
# plt.scatter(B_train[:, 0], B_train[:, 1])
# plt.show()
