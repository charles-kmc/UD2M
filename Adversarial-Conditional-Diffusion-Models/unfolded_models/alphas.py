import numpy as np
import matplotlib.pyplot as plt

betas  = np.linspace(0.0001, 0.02, 1000)
alphas = 1 - betas
alphas_cumprod = np.cumprod(alphas)

alphas_approx = np.exp(1000*(1-betas[0]) - (betas[-1] - betas[0]) * np.arange(1000)**2/2/1000)
print("alphas_approx", alphas_approx)
Ts = np.arange(1000)
def plot_alphas():
    plt.figure(figsize=(10, 5))
    plt.plot(Ts, alphas, label='alphas')
    plt.plot(Ts, alphas_cumprod, label='alphas_cumprod')
    plt.plot(Ts, alphas_approx, label='alphas_approx')
    plt.title('Alphas and Alphas Cumulative Product')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()

plot_alphas()