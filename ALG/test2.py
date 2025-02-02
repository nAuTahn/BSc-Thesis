import math
import numpy as np
import time
import json

from scipy.stats import iqr
from hmivdcma import HMIVDCMA


def main():
    dim = 50
    dim_int = dim // 2
    dim_co = dim - dim_int
    domain_int = [list(range(-10, 11)) for _ in range(dim_int)]
    order = np.random.permutation(30)
    #print(order)
    #domain_int = [list(range(-5, 6)) for _ in range(dim_int)]
    #domain_int = [[0,1] for _ in range(dim_int)]
    def sphere_int(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        return np.sum(xbar**2)
    def sphere_one_max(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        xbar[dim_co:] = np.where(xbar[dim_co:] > 0, 1.0, 0.0)
        return np.sum(xbar[:dim_co]**2) + dim_int - np.sum(xbar[dim_co:])
    def n_int_tablet(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        xbar[:dim_co] *= 100
        return np.sum(xbar**2)
    def ellipsoid_int(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        coefficients = np.array([math.pow(1e3, i / (dim - 1.)) for i in range(dim)]).reshape(-1,1)
        return np.sum((coefficients * xbar)**2)
    def reversed_ellipsoid_int(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        coefficients = np.array([math.pow(1e3, i / (dim - 1.)) for i in range(dim)]).reshape(-1,1)
        return np.sum((coefficients[dim_co:] * xbar[:dim_co])**2) + np.sum((coefficients[:dim_co] * xbar[dim_co:])**2)
    def rosenbrock(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        temp = np.array([100. * (xbar[i + 1] - xbar[i]**2)**2 + (1. - xbar[i])**2 for i in range(dim - 1)])#.reshape(-1, 1)
        return np.sum(temp)
    def different_powers(x):
        xbar = np.array(x)
        xbar = np.abs(xbar)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        power = np.array([2 + 10 * i / (dim - 1.) for i in range(dim)]).reshape(-1, 1)
        power = power[order]
        return np.sum(np.power(xbar, power))
    def rastrigin(x):
        y = x.copy()
        np.round(y[dim_co:])
        #y = y.reshape(-1)
        return 10*dim + np.sum([y[i] ** 2 - 10.0 * math.cos(2.0 * math.pi * y[i]) for i in range(dim)])
    def sphere_leading_one(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        Prod = np.array([np.prod(xbar[dim_co:i+1]) for i in range(dim_co, dim)])
        return np.sum((xbar[:dim_co])**2) + dim_int - np.sum(Prod)
    def ellipsoid_one_max(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        coefficients = np.array([math.pow(1e3, i / (dim_co - 1.)) for i in range(dim_co)]).reshape(-1,1)
        return np.sum((coefficients * xbar[:dim_co])**2) + dim_int - np.sum(xbar[dim_co:])
    def rosenbrock(x):
        xbar = np.array(x)
        xbar[dim_co:] = np.round(xbar[dim_co:])
        temp = np.array([100. * (xbar[i + 1] - xbar[i]**2)**2 + (1. - xbar[i])**2 for i in range(dim - 1)])#.reshape(-1, 1)
        return np.sum(temp)
    data = {
        "root": {}
    }
    num_succ = 0
    #log_d = np.floor(3 * np.log(dim))
    #lam = 4 + int(log_d) if int(log_d) % 2 == 0 else 5 + int(log_d)
    #lam = 4 + int(log_d)
    lam = 16 #18, 28, 48 vs 14, 24, 40
    budget = 15000
    num_succ_5 = 0
    num_succ_10 = 0
    num_succ_15 = 0
    num_evals = []
    begin = time.time()
    for i in range(100):
        np.random.seed(i)
        key = f"f_iters_{i}"
        mean = np.ones(dim) * 2.
        #mean = np.ones(dim)
        #mean[:dim_co] *= 2.
        #mean[dim_co:] *= 0.5
        sigma = 0.5
        margin = 1.0 / (lam * dim)
        vd = HMIVDCMA(func=ellipsoid_int, xmean0=mean, sigma0=sigma, domain_int=domain_int, margin=margin, lamb=lam, budget=budget, sd=i, ssa='TPA')
        is_success, x_best, f_best, eval, all_inds, all_mean, all_std, all_v, plot_iter_f, plot_iter_mean = vd.run()
        data["root"][key] = plot_iter_f
        #data["root"] = plot_iter_mean
        if is_success:
            if eval <= 5000:
                num_succ_5 += 1
            if eval <= 10000:
                num_succ_10 += 1
            if eval <= 15000:
                num_succ_15 += 1
            num_succ += 1
            num_evals.append(eval)
        print(f'Eval: {eval}        fbest: {f_best}')
        #print(f'x_best: {x_best}')
    end = time.time()

    all_mean = np.array(all_mean)
    all_std = np.array(all_std)
    all_inds = np.array(all_inds)
    all_v = np.array(all_v)
    #print('inds:', all_inds)
    #print('mean:', all_mean)
    #print('std:', all_std)
    #print(all_mean.shape)
    all_mean_list = all_mean.tolist()
    all_std_list = all_std.tolist()
    all_inds_list = all_inds.tolist()
    all_v_list = all_v.tolist()
    #data = {
    #    "root": {
    #        "all_mean": all_mean_list,
    #        "all_std": all_std_list,
    #        "all_inds": all_inds_list,
    #        "all_v": all_v_list
    #    }
    #}
    #with open('VD_power_f.json', 'w') as f:
    #    json.dump(data, f, indent=4)
    #with open('track_TPA_VD_reversed_ellipsoid_int.json', 'w') as f:
    #    json.dump(data, f, indent=4)

    #print('Success:', num_succ)
    #print('Time:', end - begin)
    #print("Number of Evals:", num_evals)
    #print("IQR:", iqr(num_evals))
    #print('Fraction:', np.median(num_evals) / dim)
    print('Success rate 5:', num_succ_5)
    print('Success rate 10:', num_succ_10)
    print('Success rate 15:', num_succ_15)
    #print("Success: {}, Average of Number of Evaluations (std): {} ({}), Median: {}".format(num_succ, np.mean(num_evals), np.std(num_evals), np.median(num_evals)))

if __name__ == '__main__':
    main()