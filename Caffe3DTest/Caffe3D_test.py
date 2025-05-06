import numpy as np
def calc_expect_func(x, kernel, stride, y):
    iptX = x["value"]
    iptK = kernel["value"]
    iptS = stride["value"]
    ch = iptX.shape[0]
    row = iptX.shape[1]
    col = iptX.shape[2]
    print(f"ORIGIN MATRIX ROW = {row} AND COL = {col} AND CHANNEL = {ch}")
    idxMat = [[[i + j*col + k*col*row for i in range(col)]for j in range(row)]for k in range(ch)]
    print(f"INDEX MATRIX OF X IS: {idxMat}")
    kernelH = iptK.shape[0]
    kernelW = iptK.shape[1]
    kernelSize = kernelH * kernelW
    rowStep = int((col - kernelW) / iptS[0] + 1)
    colStep = int((row - kernelH) / iptS[0] + 1)
    res = np.zeros((rowStep * colStep, kernelH * kernelW * ch), np.int32)
    for m in range(colStep):
        for n in range(rowStep):
            for p in range(ch):
                for l in range(kernelH):
                    for k in range(kernelW):
                        res[m*rowStep+n][p*kernelSize+l*kernelW+k] = idxMat[p][m*iptS[0]+l][n*iptS[0]+k]
    print(res)
    print(res.dtype)
    return [res, ]