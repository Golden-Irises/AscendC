import numpy as np

def calc_expect_func(weight, x, bias, y):
    print(weight)
    print(x)
    res = np.matmul(weight["value"], x["value"]) + bias["value"]
    return [res, ]
