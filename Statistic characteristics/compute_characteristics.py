import math
import numpy as np
import pandas as pd
from scipy import stats as stats

distributions = (stats.norm, (0, 1), "Normal"), (stats.cauchy, (0, 1), "Cauchy"), \
                (stats.laplace, (0, 1 / math.sqrt(2)), "Laplace"), \
                (stats.uniform, (-math.sqrt(3), 2*math.sqrt(3)), "Uniform"), (stats.poisson, [10], "Poisson")


def generate_selection(distribution_params, size):
    func, params, name = distribution_params[0], distribution_params[1], distribution_params[2]
    return func.rvs(*params, size=size), name


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


def halfsum_quartiles(selection):
    return (count_quartile(selection, 0.25) + count_quartile(selection, 0.75)) / 2


def median(selection):
    return np.median(selection)


def mean(selection):
    return np.mean(selection)


def halfsum_extreme_elems(section):
    return (min(section) + max(section)) / 2


def truncated_mean(selection):
    n = len(selection)
    start = round(n / 4 + 1)
    end = n - start
    new_selection = selection[start:end]
    return np.sum(new_selection) / (n - 2 * start)


def characteristic_calc(selection):
    mean_value = mean(selection)
    median_value = median(selection)
    z_R = halfsum_extreme_elems(selection)
    z_Q = halfsum_quartiles(selection)
    z_tr = truncated_mean(selection)
    return mean_value, median_value, z_R, z_Q, z_tr


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


def mean_dispersion_calc(selection):
    mean_value = mean(selection)
    dispersion_value = np.var(selection)
    sqrt_dis = math.sqrt(dispersion_value)
    min_mean = mean_value - sqrt_dis
    max_mean = mean_value + sqrt_dis
    return float("{0:.4f}".format(mean_value)), float("{0:.4f}".format(dispersion_value)), \
           float("{0:.4f}".format(max_mean)), float("{0:.4f}".format(min_mean)), \
           estimation(max_mean, min_mean)


def research(repetitions=1000, sizes=[10, 100, 1000]):
    results = dict()
    for distribution in distributions:
        for size in sizes:
            mean_values = list()
            median_values = list()
            z_Rs = list()
            z_Qs = list()
            z_trs = list()
            for i in range(repetitions):
                selection, name = generate_selection(distribution, size)
                mean_value, median_value, z_R, z_Q, z_tr = characteristic_calc(selection)
                mean_values.append(mean_value)
                median_values.append(median_value)
                z_Rs.append(z_R)
                z_Qs.append(z_Q)
                z_trs.append(z_tr)
            results[name + str(size)] = [mean_dispersion_calc(mean_values), mean_dispersion_calc(median_values),
                                         mean_dispersion_calc(z_Rs), mean_dispersion_calc(z_Qs),
                                         mean_dispersion_calc(z_trs)]
    for key in results:
        write_csv(results, key)


def write_csv(res, tablename):
    results = pd.DataFrame(res[tablename]).T
    results.rename(columns={0: "mean(x)", 1: "median(x)", 2: "zR", 3: "zQ", 4: "ztr"}, inplace=True)
    results.index = ["E(x)", "D(x)", "E + sqrt(D)", "E - sqrt(D)", "Estimation"]
    results.to_csv(tablename + ".csv", sep=",")
    print(results)


# main
research()
#print(estimation(0.3045, -0.3304))

