import numpy as np

def calc_expect_func(x, poolSize, y):
    print(f"THIS IS INPUT MATRIX AFTER IM2COL: {x}")
    print(f"THIS IS POOL WINDOW SIZE: {poolSize}")
    iptMat = x["value"]
    iptPool = poolSize["value"]
    row = iptMat.shape[0]
    col = (int)(iptMat.shape[1] / iptPool[0])
    res = np.zeros((row, col), np.float32)
    for i in range(row):
        for j in range(col):
            for k in range(iptPool[0]):
                res[i][j] += iptMat[i][j*iptPool[0] + k]
            res[i][j] /= iptPool[0]
    print(res)
    print(res.dtype)
    return [res, ]