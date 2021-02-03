import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as stats


def generate_distribution(distribution_params, size):
    func, params, name = distribution_params[0], distribution_params[1], distribution_params[2]

    def get_probability_func(x):
        return func.pdf(x, *params)

    return func.rvs(*params, size), get_probability_func, name


def generate_poisson_distribution(size):
    func, param, name = stats.poisson, 10, "Poisson"

    def get_probability_func(k):
        return func.pmf(k, mu=param)

    return func.rvs(mu=param, size=size), get_probability_func, name


def all_distributions(sizes):
    distributions = (stats.norm, (0, 1), "Normal"), (stats.cauchy, (0, 1), "Cauchy"), \
                    (stats.laplace, (0, 1 / math.sqrt(2)), "Laplace"), \
                    (stats.uniform, (-math.sqrt(3), math.sqrt(3)), "Uniform")
    for distribution in distributions:
        for size in sizes:
            selection, probability_func, name = generate_distribution(distribution, size)
            yield selection, probability_func, name, size


def x_axes(min_x, max_x, step):
    return np.arange(min_x, max_x + step, step)


def search_borders(selection):
    min_x = math.floor(min(selection))
    max_x = math.ceil(max(selection))
    if max_x - min_x > 100:
        while max_x - min_x > 100:
            min_x /= 2
            max_x /= 2
    return min_x, max_x


def create_bins(min_x, max_x, step):
    return math.ceil((max_x - min_x) / step)


def y_axes(x_axes, func):
    axes = list()
    for ax in x_axes:
        axes.append(func(ax))
    return axes


def draw_histograms_and_functions(selection, step, func, name, size):
    min_x, max_x = search_borders(selection)
    x_values = x_axes(min_x, max_x, step)
    y_values = y_axes(x_values, func)
    label = str(name) + " distribution " + str(size) + " elements"
    plt.hist(selection, range=(min_x, max_x), align="mid", rwidth=0.95, color='m', density=True,
             bins=create_bins(min_x, max_x, 1))
    plt.plot(x_values, y_values)
    plt.xlabel("Numbers")
    plt.ylabel("Density")
    plt.title(label)
    plt.show()


def build_histograms(sizes):
    step = 0.1
    for data in all_distributions(sizes):
        selection, func, name, size = data
        draw_histograms_and_functions(selection, step, func, name, size)
    step = 1
    for size in sizes:
        selection, func, name = generate_poisson_distribution(size)
        draw_histograms_and_functions(selection, step, func, name, size)


# Main function
build_histograms([10, 50, 1000])
