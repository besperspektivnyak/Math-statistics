import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as stats

distributions = (stats.norm, (0, 1), "Normal"), (stats.cauchy, (0, 1), "Cauchy"), \
                (stats.laplace, (0, 1 / math.sqrt(2)), "Laplace"), \
                (stats.uniform, (-math.sqrt(3), 2 * math.sqrt(3)), "Uniform"), (stats.poisson, [10], "Poisson")


def generate_selection(distribution_params, size):
    func, params, name = distribution_params[0], distribution_params[1], distribution_params[2]
    return np.array(func.rvs(*params, size=size)), name


def is_int(n):
    if int(n) == float(n):
        return True
    else:
        return False


def count_quartile(selection, order):
    selection.sort()
    length = len(selection)
    tmp = length * order
    if is_int(tmp):
        return selection[int(tmp)]
    else:
        return selection[int(tmp) + 1]


def count_borders(selection):
    x_1 = count_quartile(selection, 0.25) - 3 / 2 * (count_quartile(selection, 0.75) - count_quartile(selection, 0.25))
    x_2 = count_quartile(selection, 0.75) + 3 / 2 * (count_quartile(selection, 0.75) - count_quartile(selection, 0.25))
    return x_1, x_2


def find_outliers(selection):
    left_border, right_border = count_borders(selection)
    outliers = list()
    for element in selection:
        if element < left_border or element > right_border:
            outliers.append(element)
    return outliers


def count_share_outliers(selection):
    outliers = find_outliers(selection)
    share = len(outliers) / len(selection)
    return share


def draw_boxplot(sample, name):
    plt.boxplot(sample, sym="o", labels=["n=20", "n=100"], vert=False)
    plt.title(name)
    plt.savefig(name + '.png')
    plt.show()
    plt.boxplot(sample, sym="o", labels=["n=20", "n=100"], vert=False)
    plt.title(name)
    plt.xlim([-10, 10])
    plt.savefig(name + '_zoomed' + '.png')
    plt.show()


def build_boxplots(sizes=[20, 100]):
    for distribution in distributions:
        selections = list()
        for size in sizes:
            selections.append(generate_selection(distribution, size)[0])
        draw_boxplot(tuple(selections), distribution[2])


def estimation_for_big_num(max_value, min_value):
    if int(max_value) == int(min_value):
        return int(max_value)
    else:
        return "-"


def estimation(max_value, min_value):
    if max_value > 1:
        return estimation_for_big_num(max_value, min_value)
    value = max_value - min_value
    cases = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 1]
    if cases[0] < value:
        return int(0)
    elif cases[0] > value > cases[1]:
        return ('%.2f' % max_value)[:-1]
    elif cases[1] > value > cases[2]:
        return ('%.3f' % max_value)[:-1]
    elif cases[2] > value > cases[3]:
        return ('%.4f' % max_value)[:-1]
    elif cases[3] > value > cases[4]:
        return ('%.5f' % max_value)[:-1]
    elif cases[4] > value > cases[5]:
        return ('%.6f' % max_value)[:-1]


def mean_calc(selection):
    mean_value = np.mean(selection)
    dispersion_value = np.var(selection)
    sqrt_dis = math.sqrt(dispersion_value)
    min_mean = mean_value - sqrt_dis
    max_mean = mean_value + sqrt_dis
    return estimation(max_mean, min_mean)


def calculate_outliers(sizes=[20, 100], repetitions=1000):
    result = dict()
    for distribution in distributions:
        for size in sizes:
            shares = list()
            for i in range(repetitions):
                selection = generate_selection(distribution, size)
                shares.append(count_share_outliers(selection[0]))
            result[distribution[2] + str(size)] = mean_calc(shares)
    print(result)
    write_to_csv(result)


def write_to_csv(result):
    results = pd.DataFrame(result.items())
    results.rename(columns={0: "Selection", 1: "Share of outliers"}, inplace=True)
    print(results)
    results.to_csv("outliers.csv", sep=",")


# main
build_boxplots()
calculate_outliers()