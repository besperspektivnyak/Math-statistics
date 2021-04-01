import math as math
import scipy.stats as stats
from scipy.optimize import minimize
import pandas as pd


# def lik(parameters, x):
#     mu = parameters[0]
#     sigma = parameters[1]
#     n = len(x)
#     L = n / 2.0 * np.log(2 * np.pi) + n / 2.0 * math.log(sigma ** 2) + 1 / (2 * sigma ** 2) * \
#         sum([(x_ - mu)**2 for x_ in x])
#     return L


def mle(func, sample):
    mle_func = lambda params: -sum([math.log(math.fabs(func(x_i, *params))) for x_i in sample])

    return minimize(mle_func, [1, 1]).x


def grid_generator(n):
    #k = math.ceil(1.72* (n**(1/3)))
    k = 5
    l_b, r_b = -1.1, 1.1
    step = (r_b-l_b)/k
    res = [(-math.inf, l_b)]
    res += [(l_b + step*i, l_b+step*(i+1)) for i in range(k)]
    res += [(r_b, math.inf)]
    return res


def get_n_i(sample):
    def n_i(col):
        new_col = []
        c = 0

        for cut in col:
            new_col.append(0)
            for i in sample:
                new_col[c] += 0 if (i <= cut[0] or i >= cut[1]) else 1
            c += 1
        return new_col

    return n_i


def get_p_i(F):
    def p_i(col):
        new_col = []
        for cut in col:
            new_col.append(F(cut[1]) - F(cut[0]))
        return new_col

    return p_i


def create_table(sample, F, alpha=0.05):
    ni = get_n_i(sample)
    pi = get_p_i(F)
    frame = pd.DataFrame(columns=['Borders'])
    frame['Borders'] = grid_generator(len(sample))
    frame['$n_i$'] = ni(frame['Borders'])
    frame['$p_i$'] = pi(frame['Borders'])
    frame['$np_i$'] = len(sample) * frame['$p_i$']
    frame['$n_i-np_i$'] = frame['$n_i$'] - frame['$np_i$']
    frame['$(n_i - np_i)^2/(np_i)$'] = (frame['$n_i$'] - frame['$np_i$']) ** 2 / (frame['$np_i$'])
    frame = frame.round(4)
    print(frame)
    print(sum(frame['$(n_i - np_i)^2/(np_i)$']))
    frame.to_csv('3.csv')


# sample = np.array(stats.norm.rvs(0, 1, size=100))
sample = stats.laplace.rvs(0, 1, size = 20)

t, y = mle(stats.norm.pdf, sample)
hypotise = lambda x: stats.norm.cdf(x, t, y)

print(t, y)

create_table(sample, hypotise)