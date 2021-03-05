import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as stats


distributions = (stats.norm, (0, 1), "Normal", [-4, 4]), (stats.cauchy, (0, 1), "Cauchy", [-4, 4]), \
                (stats.laplace, (0, 1 / math.sqrt(2)), "Laplace", [-4, 4]), \
                (stats.uniform, (-math.sqrt(3), 2 * math.sqrt(3)), "Uniform", [-4, 4]), \
                (stats.poisson, [10, 0], "Poisson", [6, 14])


def generate_distribution(distribution_params, size):
    func, params, name, borders = distribution_params[0], distribution_params[1], distribution_params[2], \
                                  distribution_params[3]

    def density_func(x):
        return func.cdf(x, *params)

    return func.rvs(*params, size), density_func, name, borders


def ecdf(selection):
    selection = np.sort(selection)

    def result(x):
        return np.searchsorted(selection, x, side='right') / selection.size

    return result


def get_points(borders, func, step):
    x_values = np.arange(borders[0], borders[1], step)
    y_values = list(map(func, x_values))
    return x_values, y_values


def draw_cdf(functions, borders, name, size):
    fig, ax = plt.subplots()
    for function in functions:
        ax.plot(*get_points(borders, function, step=0.05))
    ax.set(xlabel='values', ylabel='f(x)', title=name +' n = ' + str(size))
    ax.grid()
    plt.legend(['Theoretical func', 'Empirical func'])

    fig.savefig(name + str(size) + ".png")
    plt.show()


def research_cdf(sizes=[20, 60, 100]):
    for distribution in distributions:
        for size in sizes:
            selection, density_func, name, borders = generate_distribution(distribution, size)
            e_cdf = ecdf(selection)
            draw_cdf([density_func, e_cdf], borders, name, size)


# research_cdf()
