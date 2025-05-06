import numpy as np
def calc_expect_func(x, kernel, stride, y):
    iptX = x["value"]
    iptK = kernel["value"]
    iptS = stride["value"]
    row = iptX.shape[0]
    col = iptX.shape[1]
    print(f"ORIGIN MATRIX ROW = {row} AND COL = {col}")
    idxMat = [[i + j*col for i in range(col)]for j in range(row)]
    print(f"INDEX MATRIX OF X IS: {idxMat}")
    kernelH = iptK.shape[0]
    kernelW = iptK.shape[1]
    rowStep = int((col - kernelW) / iptS[0] + 1)
    colStep = int((row - kernelH) / iptS[0] + 1)
    res = np.zeros((rowStep * colStep, kernelH * kernelW), np.int32)
    for m in range(colStep):
        for n in range(rowStep):
            for l in range(kernelH):
                for k in range(kernelW):
                    res[m*rowStep+n][l*kernelW + k] = idxMat[m*iptS[0]+l][n*iptS[0]+k]
    print(res)
    print(res.dtype)
    return [res, ]
