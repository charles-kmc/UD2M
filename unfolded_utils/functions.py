
import numpy as np

def chebyshev(j, x, derivative = False):
    x = np.array(x)
    if not derivative:
        if j == 0:
            return np.ones(x.shape)
        elif j==1:
            return x
        else: 
            Ts = [1, x] 
            for j in range(2, j+1):
                Ts.append(2*x*Ts[-1] - Ts[-2]) 
        return Ts[-1]
    
    else: 
        if j == 0:
            return np.zeros(x.shape)
        elif j == 1:
            return np.ones(x.shape)
        else: 
            return (
                2*chebyshev(j-1, x, derivative=False) 
                + 2*x*chebyshev(j-1, x, derivative = True) 
                - chebyshev(j-2, x, derivative = True)
            )
        

    
if __name__ == '__main__':
    import numpy as np 
    import matplotlib.pyplot as plt 
    x = np.linspace(-1,1,1000)
    js = [0,1,2,3]
    plt.figure()
    for j in js:
        T = chebyshev(j, x, derivative = True)
        plt.plot(x, T)
    plt.show()