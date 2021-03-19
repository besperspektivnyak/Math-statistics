import numpy as np
from scipy import stats as stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def build_grid(begin, end, step):
    return np.arange(begin, end, step)


def func(x):
    return 2 * x + 2


def get_error(loc, scale):
    return stats.norm.rvs(loc, scale)


def noisy_func(xs):
    y = list()
    for x in xs:
        y.append(func(x) + get_error(0, 1))
    return y


def OLS(x, y):
    regression = LinearRegression().fit(x, y)
    b_0 = regression.intercept_
    b_1 = regression.coef_
    y_new = [b_0 + b_1 * x_ for x_ in x]
    print(b_0, b_1)
    return y_new


def LAD(x, y):
    y_new = [y_ - get_error(0, 1) for y_ in y]
    regression = LinearRegression().fit(x, y_new)
    b_0 = regression.intercept_
    b_1 = regression.coef_
    y_new = [b_0 + b_1 * x_ for x_ in x]
    print(b_0, b_1)
    return y_new


x = build_grid(-1.8, 2.2, 0.2).reshape(-1, 1)
x_ = build_grid(-1.8, 2.2, 0.2)
y = noisy_func(x_)
y[0] += 10
y[19] -= 10
plt.plot(build_grid(-1.8, 2.2, 0.2), func(x), label="Модель")
plt.plot(build_grid(-1.8, 2.2, 0.2), OLS(x, y), label="МНК")
plt.plot(build_grid(-1.8, 2.2, 0.2), LAD(x, y), label="МНМ")
plt.scatter(build_grid(-1.8, 2.2, 0.2), y, c='red', label='Выборка')
plt.legend()
# plt.save  fig('2.png')
plt.show()

x = build_grid(-1.8, 2.2, 0.2).reshape(-1, 1)
y = noisy_func(x)
OLS(x, y)
LAD(x, y)

