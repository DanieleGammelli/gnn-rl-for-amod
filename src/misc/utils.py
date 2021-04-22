import numpy as np

def mat2str(mat):
    return str(mat).replace("'",'"').replace('(','<').replace(')','>').replace('[','{').replace(']','}')  

def dictsum(dic,t):
    return sum([dic[key][t] for key in dic if t in dic[key]])

def moving_average(a, n=3) :
    """
    Computes a moving average used for reward trace smoothing.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n