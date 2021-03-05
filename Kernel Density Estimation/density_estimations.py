import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as stats


distributions = (stats.norm, (0, 1), "Normal", [-4, 4]), (stats.cauchy, (0, 1), "Cauchy", [-4, 4]), \
                (stats.laplace, (0, 1 / math.sqrt(2)), "Laplace", [-4, 4]), \
                (stats.uniform, (-math.sqrt(3), 2 * math.sqrt(3)), "Uniform", [-4, 4]),\
                (stats.poisson, [10, 0], "Poisson", [6, 14])

# distributions = [(stats.poisson, [10, 0], "Poisson", [6, 14])]


def generate_distribution(distribution_params, size):
    func, params, name, borders = distribution_params[0], distribution_params[1], distribution_params[2], \
                                  distribution_params[3]

    if name == "Poisson":
        def get_probability_func(k):
            return func.pmf(k, *params)

    else:
        def get_probability_func(x):
            return func.pdf(x, *params)

    return func.rvs(*params, size), get_probability_func, name, borders


def get_points(borders, func, step):
    x_values = np.arange(borders[0], borders[1], step)
    y_values = list(map(func, x_values))
    return x_values, y_values


def h(selection):
    return 1.06 * np.std(selection) * (len(selection) ** (-1 / 5))


def draw_estimation(functions, borders, name, size, k):
    fig, ax = plt.subplots()
    for function in functions:
        ax.plot(*get_points(borders, function, step=1))
    ax.set(xlabel='values', ylabel='f(x)', title=name +' n = ' + str(size))
    ax.grid()
    plt.legend(['Probability density function', 'Kernel density estimation'])

    fig.savefig(name + str(size) + "h=" + str(k) + "Est.png")
    plt.show()


def research_estimation(sizes=[20, 60, 100]):
    k = [1/2, 1, 2]
    for distribution in distributions:
        for size in sizes:
            selection, probability_func, name, borders = generate_distribution(distribution, size)
            for coeff in k:
                kde = stats.gaussian_kde(selection)
                kde.set_bandwidth(bw_method=h(selection) * coeff / 4)
                draw_estimation([probability_func, kde.pdf], borders, name, size, coeff)
                print(str(h(selection) * coeff / 4) + name + str(size) + ' ' + str(coeff))


research_estimation()