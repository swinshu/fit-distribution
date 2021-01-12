# import
import numpy as np
from numpy import random
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt

data = random.normal(0, 2, 100)
print(data)

mean, dev = stats.distributions.norm.fit(data)
x = np.linspace(norm.ppf(0.01, loc = 0.0, scale = 2.0), norm.ppf(0.99, loc = 0.0, scale = 2.0), 100)

fitted = stats.distributions.norm.pdf(x, mean, dev)

plt.hist(data, density = True)
plt.plot(x, fitted, 'r-')

plt.show()