import pandas as pd
import numpy as np
import warnings
from scipy import stats as st
from scipy.stats import norm
import matplotlib.pyplot as plt

file = '/Users/michellesun/Documents/dist_file.xlsx'
data = pd.read_excel(file, index_col = 0)

mean, dev = st.distributions.norm.fit(data)
x = np.linspace(norm.ppf(0.01, loc = mean, scale = dev), norm.ppf(0.99, loc = mean, scale = dev), 10000)

y, z = np.histogram(data, bins = 200, density = True)

def best_fit(data, bins = 200, ax = None):
    # distributions to check
    DISTRIBUTIONS = [
            st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
            st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
            st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
            st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
            st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
            st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
            st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
            st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
            st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
            st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
        ]

    # best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # estimate distribution parameters from data
    for dist in DISTRIBUTIONS:
        # try fitting
        try:
            # ignore warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                # fit distribution to data
                params = dist.fit(data)
                # separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                var = params[-1]

                # calculate fitted pdf and error with fit in distribution
                pdf = st.distributions.pdf(x, loc = loc, scale = var, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
            # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax = ax)
                except Exception:
                    pass
            # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = dist
                    best_params = params
                    best_sse = sse
        except Exception:
            pass

    return (best_distribution.name, best_params)


def make_pdf(dist, params, size = 10000):
    # separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    var = params[-1]

    # get start and end points of distribution
    start = dist.ppf(0.01, *arg, loc = loc, scale = var) if arg else dist.ppf(0.01, loc = loc, scale = var)
    end = dist.ppf(0.99, *arg, loc = loc, scale = var) if arg else dist.ppf(0.99, loc = loc, scale = var)

    # build pdf and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc = loc, scale = var, *arg)
    pdf = pd.Series(y, x)

    return pdf

plt.figure()
ax = data.plot(kind = 'hist', bins = 25)
y_lim = ax.get_ylim()

# find best fit distribution
best_fit_name, best_fit_params = best_fit(data, 200, ax)
best_dist = getattr(st, best_fit_name)

print(best_fit_name)
# update plots
ax.set_ylim(y_lim)

# make pdf with best params
pdf = make_pdf(best_dist, best_fit_params)
print(pdf)
pdf.plot(kind = 'hist', bins = 25, alpha = 0.5, legend = False)

mean, dev = st.distributions.norm.fit(data)
# x = np.linspace(norm.ppf(0.01, loc = mean, scale = dev), norm.ppf(0.99, loc = mean, scale = dev), 10000)
plt.plot(x, pdf, 'r-')
plt.show()

fitted = st.distributions.norm.pdf(x, mean, dev)

# plt.hist(data, density = True)
# plt.plot(x, fitted, 'r-')
# plt.show()