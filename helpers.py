import numpy as np

def find_n(N,T):
    left = N
    right = N/(1-np.exp(-1))
    
    def search_n(l,r,T):
        f = lambda x: x*(1-np.exp(-T/x))
        while(f(r) < N):
            l = r
            r = r * 1.5
        n = (l+r)/2
        while(np.abs(f(n)-N) > 1):
            if f(n) - N < 0:
                l = n
                n = (l+r)/2
            else:
                r = n
                n = (l+r)/2
        return n
    return round(search_n(left,right,T)).astype(int)


def plot_CDF_helper(data):
    h, edges = np.histogram(data, density=True, bins=100,)
    h = np.cumsum(h)/np.cumsum(h).max()

    X = edges.repeat(2)[:-1]
    y = np.zeros_like(X)
    y[1:] = h.repeat(2)
    return X,y 
