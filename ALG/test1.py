import math
import numpy as np
import copy
from scipy.stats import norm
from scipy.stats import chi2
import time
from bisect import bisect_left
from scipy.stats import iqr
import json
from hmifmnes import HMIFMNES

def int_feasible(dim, interval):
    feasible_int_var_2dlist = [
        [0, 1],
        [0, 1, 2, 3],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [0]
    ]

    int_var_2dlist = []
    counter = 1
    for i in range(dim - interval):
        int_var_2dlist.append(feasible_int_var_2dlist[counter-1])
        if i + 1 >= counter * interval:
            counter += 1
    int_var_2dlist = list(int_var_2dlist)
    return int_var_2dlist


def main():
    dim = 50
    dim_int = dim // 2
    dim_co = dim - dim_int
    domain_int = [list(range(-10, 11)) for _ in range(dim_int)]
    #domain_int = [list(range(-5, 6)) for _ in range(dim_int)]
    #domain_int = [[0,1] for _ in range(dim_int)]

    def n_int_tablet(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        xbar[:dim_co] *= 100
        return np.sum(xbar**2)
    def ellipsoid_int(x):
        #xt = x - 9.
        xbar = np.array(x)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        coefficients = np.array([math.pow(1e3, i / (dim - 1.)) for i in range(dim)]).reshape(-1,1)
        return np.sum((coefficients * xbar)**2)
    def reversed_ellipsoid_int(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        coefficients = np.array([math.pow(1e3, i / (dim - 1.)) for i in range(dim)]).reshape(-1,1)
        return np.sum((coefficients[dim_co:] * xbar[:dim_co])**2) + np.sum((coefficients[:dim_co] * xbar[dim_co:])**2)
    def sphere_one_max(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        #xbar[dim_co:] = np.where(xbar[dim_co:] > 0, 1.0, 0.0)
        return np.sum(xbar[:dim_co]**2) + dim_int - np.sum(xbar[dim_co:])
    def sphere_int(x):
        #xt = x.copy() - 9.
        xbar = np.array(x)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        return np.sum(xbar**2)
    def ellipsoid_one_max(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        coefficients = np.array([math.pow(1e3, i / (dim_co - 1.)) for i in range(dim_co)]).reshape(-1,1)
        return np.sum((coefficients * xbar[:dim_co])**2) + dim_int - np.sum(xbar[dim_co:])
    def sphere_leading_one(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        Prod = np.array([np.prod(xbar[dim_co:i+1]) for i in range(dim_co, dim)])
        return np.sum((xbar[:dim_co])**2) + dim_int - np.sum(Prod)
    def ellipsoid_leading_one(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        coefficients = np.array([math.pow(1e3, i / (dim_co - 1.)) for i in range(dim_co)]).reshape(-1,1)
        Prod = np.array([np.prod(xbar[dim_co:i+1]) for i in range(dim_co, dim)])
        return np.sum((coefficients * xbar[:dim_co])**2) + dim_int - np.sum(Prod)
    def rastrigin(x):
        y = x.copy()
        np.round(y[dim_co:])
        #y = y.reshape(-1)
        return 10*dim + np.sum([y[i] ** 2 - 10.0 * math.cos(2.0 * math.pi * y[i]) for i in range(dim)])
    def rosenbrock(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        temp = np.array([100. * (xbar[i + 1] - xbar[i]**2.)**2. + (1. - xbar[i])**2. for i in range(dim - 1)])#.reshape(-1, 1)
        return np.sum(temp)
    def different_powers(x):
        xbar = np.array(x)
        xbar = np.abs(xbar)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        power = np.array([2 + 10 * i / (dim - 1.) for i in range(dim)])#.reshape(-1, 1)
        return np.sum(np.power(xbar, power))
    #data = {
    #    "root": {}
    #}
    log_d = np.floor(3 * np.log(dim))
    num_ac = 0
    num_evals = []
    #budget = dim * 10 * 10 * 10 * 10
    budget = 15000
    target = 1e-10
    num_ac_5 = 0
    num_ac_10 = 0
    num_ac_15 = 0
    start = time.time()
    for i in range(100):
        #key = f"mean_evals_{i}"
        mean = np.ones([dim, 1]) * 2.
        #mean = np.ones([dim, 1])
        #mean[:dim_co] *= 2.
        #mean[dim_co:] *= 0.5
        lamb = 16
        sigma = 0.5
        #lamb = 4 + int(log_d) if int(log_d) % 2 == 0 else 5 + int(log_d)
        #print(lamb)
        list_margin = np.array([1.0 / (dim * lamb) for _ in range(dim_int)])
        #print('random:', np.random.randint(0, 2, dim_co))
        other_margin = np.array([2.0 / (dim * lamb), 1.0 / (dim * lamb), 1. / (2 * dim * lamb)])
        margin = 1.0 / (dim * lamb)
        lmi = HMIFMNES(dim_int=dim_int, dim_co=dim_co, f=reversed_ellipsoid_int, m=mean, sigma=sigma, lamb=lamb, domain_int=domain_int, margin=margin, list_margin=list_margin, other_margin=other_margin)
        succ, x_best, f_best, evals, all_mean, all_std, all_inds, all_v = lmi.optimize(budget, target, 100000)
        print(f'Eval: {evals}        fbest: {f_best}')
        #print(f'x_best: {x_best}')
        #print(x_best)
        #data["root"] = plot_mean_eval
        #print(f_best)
        if succ:
            if evals <= 5000:
                num_ac_5 += 1
            if evals <= 10000:
                num_ac_10 += 1
            if evals <= 15000:
                num_ac_15 += 1
            num_ac += 1
            num_evals.append(evals)
    end = time.time()
    #with open('sphere_int.json', 'w') as f:
    #    json.dump(data, f, indent=4)

    #print('std:', all_std)
    all_mean = np.array(all_mean)
    all_std = np.array(all_std)
    all_inds = np.array(all_inds)
    all_v = np.array(all_v)
    print(all_mean.shape)
    all_mean_list = all_mean.tolist()
    all_std_list = all_std.tolist()
    all_inds_list = all_inds.tolist()
    all_v_list = all_v.tolist()
    data = {
        "root": {
            "all_mean": all_mean_list,
            "all_std": all_std_list,
            "all_inds": all_inds_list,
            "all_v": all_v_list
        }
    }
    #with open('fmnes_demo.json', 'w') as f:
    #    json.dump(data, f, indent=4)

    #np.sort(num_evals)
    #print("Time:", (end - start))
    #print("Number of Evals:", num_evals)
    #print("IQR:", iqr(num_evals))
    #print('Fraction:', np.median(num_evals) / dim)
    print("Success rate 5: {}".format(num_ac_5))
    print("Success rate 10: {}".format(num_ac_10))
    print("Success rate 15: {}".format(num_ac_15))
    #print("Success: {}, Average of Number of Evaluations (std): {} ({}), Median: {}".format(num_ac, np.mean(num_evals), np.std(num_evals), np.median(num_evals)))


if __name__ == '__main__':
    main()