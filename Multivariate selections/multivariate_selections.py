import numpy as np
from scipy import stats as stats
import pandas as pd


def generate_selections(correlation, mean_x, mean_y, std_x, std_y, size):
    cov = correlation * std_x * std_y
    return stats.multivariate_normal(mean=[mean_x, mean_y], cov=[[std_x ** 2, cov], [cov, std_y ** 2]]).rvs(size=size)


def generate_complex_selection(correlation1, mean_x1, mean_y1, std_x1, std_y1, correlation2, mean_x2, mean_y2, std_x2,
                               std_y2, size):
    selection_1 = generate_selections(correlation1, mean_x1, mean_y1, std_x1, std_y1, size)
    selection_2 = generate_selections(correlation2, mean_x2, mean_y2, std_x2, std_y2, size)

    return 0.9 * selection_1 + 0.1 * selection_2


def mean_quad(selection):
    new_selection = list()
    for element in selection:
        new_selection.append(element ** 2)
    return np.mean(np.array(new_selection))


def correlation_pearson(selection):
    return stats.pearsonr(*(selection.transpose()))[0]


def correlation_spearman(selection):
    return stats.spearmanr(selection)[0]


def columns(sample):
    return sample[:, 0], sample[:, 1]


def get_quadrant_counter(x1, y1, offset=(0, 0)):
    functions = (
        lambda x, y: x > offset[0] and y > offset[1],
        lambda x, y: x < offset[0] and y > offset[1],
        lambda x, y: x < offset[0] and y < offset[1],
        lambda x, y: x > offset[0] and y < offset[1]
    )

    def counter(quadrant_n):
        nonlocal functions
        nonlocal x1, y1
        c = 0

        for i, j in zip(x1, y1):
            if (functions[quadrant_n - 1])(i, j):
                c += 1
        return c

    return counter


def correlation_quad(selection):
    c = get_quadrant_counter(*columns(selection), (np.median(selection[:, 0]), np.median(selection[:, 1])))

    return ((c(1) + c(3)) - (c(2) + c(4))) / len(selection)


def calculate_characteristics(selection):
    return correlation_pearson(selection), correlation_spearman(selection), correlation_quad(selection)


def compute(selection):
    return np.mean(selection), mean_quad(selection), np.var(selection)


def research_complex(sizes=[20, 60, 100], repetitions=1000):
    results = dict()
    for size in sizes:
        dispersion_pearson = list()
        dispersion_spearman = list()
        dispersion_quad = list()
        for r in range(repetitions):
            selection = generate_complex_selection(0.9, 0, 0, 1, 1, -0.9, 0, 0, 10, 10, 10)
            pearson, spearman, quad = calculate_characteristics(selection)
            dispersion_pearson.append(pearson)
            dispersion_spearman.append(spearman)
            dispersion_quad.append(quad)
        results['complex' + str(size)] = [compute(dispersion_pearson), compute(dispersion_spearman),
                                          compute(dispersion_quad)]
        for key in results:
            write_csv(results, key)


def research(sizes=[20, 60, 100], repetitions=1000):
    results = dict()
    corr_coeff = [0, 0.5, 0.9]
    for c in corr_coeff:
        for size in sizes:
            dispersion_pearson = list()
            dispersion_spearman = list()
            dispersion_quad = list()
            for r in range(repetitions):
                selection = generate_selections(c, 0, 0, 1, 1, size)
                pearson, spearman, quad = calculate_characteristics(selection)
                dispersion_pearson.append(pearson)
                dispersion_spearman.append(spearman)
                dispersion_quad.append(quad)
            results[str(size) + 'rho' + str(c)] = [compute(dispersion_pearson), compute(dispersion_spearman),
                                                    compute(dispersion_quad)]
            for key in results:
                write_csv(results, key)


def write_csv(res, tablename):
    results = pd.DataFrame(res[tablename]).T
    results.rename(columns={0: "$r$", 1: "$r_S$", 2: "$r_Q$"}, inplace=True)
    results.index = ["$E(x)$", "$E(x^2)$", "$D(x)$"]
    results.to_csv(tablename + ".csv", sep=",", float_format="%.3f")


research()
research_complex()
