import numpy as np

np.random.seed(0)
# This code implements the RandProx algorithm for Gaussian Inverse Problem
a = 0.1
mu = 10

sigma_y = 0.05
sigma_x = 1.

rho = 1*sigma_x*sigma_y/(sigma_x**2 + sigma_y**2)**0.5

def proxf(x, y, a = a, mu =mu, rho = rho, sigma_y = sigma_y):
    return (a**2/sigma_y**2 + 1/rho**2)**(-1) * (a*y/sigma_y**2 + x/rho**2)


def MMSEDen(z, rho, sigma_x = sigma_x, mu = mu):
    return (sigma_x**2*z + rho**2*mu)/(sigma_x**2 + rho**2)

def RandProx(x0, y, steps, a = a, mu = mu, rho = rho):
    out = {
        "x":[],
        "z":[],
    }
    x = x0
    for i in range(steps):
        z = proxf(x, y) + rho*np.random.normal(0, 1)
        x = MMSEDen(z, rho)
        out["x"].append(x)
        out["z"].append(z)
    return out




### Run RandProx for 1000 iterations and plot histogram of x
import matplotlib.pyplot as plt

x = mu + sigma_x*np.random.normal(0, 1)
y = a*x + sigma_y*np.random.normal(0, 1)

steps = 1000000
x0 = y/a


xs = np.linspace(5, 15, 100)
mu_true = (sigma_x**2*a*y + sigma_y**2*mu)/(sigma_x**2*a**2 + sigma_y**2)
sigma_sq_true = (sigma_x**2*sigma_y**2)/(sigma_x**2*a**2 + sigma_y**2)
true_density = np.exp(-0.5*(xs - mu_true)**2/sigma_sq_true)
true_density /= np.sum(true_density) *(xs[1] - xs[0])
plt.plot(xs, true_density, label="True Density")

out = RandProx(x0, y, steps)
plt.hist(out["x"], bins=50, density=True)
plt.title("Histogram of x")
plt.xlabel("x")
plt.ylabel("Density")

plt.figure()
plt.plot((np.array(out["x"][:-1])-np.array(out["x"][1:]))**2)
plt.show()

