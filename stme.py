import numpy as np
from numpy.lib.arraysetops import isin
from scipy.stats._continuous_distns import genpareto
from statsmodels.distributions.empirical_distribution import ECDF
from math import log


def _f_hat_cdf(pd_nrm, pd_ext, X):
    X = np.asarray(X)
    scalar_input = False
    if X.ndim == 0:
        X = X[None]  # Makes x 1D
        scalar_input = True
    val = np.zeros(X.shape)
    mu = pd_ext.args[1]  # args -> ((shape, loc, scale),)
    for i, x in enumerate(X):
        if x > mu:
            val[i] = 1 - (1 - pd_nrm(mu)) * (1 - pd_ext.cdf(x))
        else:
            val[i] = pd_nrm(x)
    if scalar_input:
        return np.squeeze(val)
    return val


def gumbel_transform(x, thr):
    """
    data   : Vector of stme values (Number of Events,1)
    t      : Scalar of threshold for each variable

    Outputs:
    X_gum  : Variables of X in gumbel scale
    """
    assert isinstance(x, np.ndarray)
    # create ecdf
    ecdf = ECDF(x)

    # fit extremes of Y to generalized pareto dist
    xp, mp, sp = genpareto.fit(x[x > thr], floc=thr)
    gp = genpareto(xp, mp, sp)

    # transform to gumbel
    x_g = -np.log(-np.log(_f_hat_cdf(ecdf, gp, x)))

    # pass function
    func = lambda x: _f_hat_cdf(ecdf, gp, x)

    return x_g, gp, func


def ts_sample(N, pool_size, ts):
    from random import randint

    assert pool_size == sum(ts)  # check if event num in pool == event num in ts
    # pool(477,2)
    samples = np.full((N, pool_size), False)
    for i in range(N):
        for j in range(len(ts)):
            k = sum(ts[0:j]) + randint(0, ts[j] - 1)
            samples[i, k] = True
    return samples


def cost(p, data_conditioning, data_conditioned):
    """
    cost(p,data,vi)->float
    p: parameter; [a,b,mu,sigma]
    data: ndarray with shape(#ofEvent, #ofVar)
    vi: Index of extreme variable
    minimize this.
    """
    data_conditioning = np.asarray(data_conditioning)
    data_conditioned = np.asarray(data_conditioned)

    q = 0
    a = p[0]
    b = p[1]
    mu = p[2]
    sigma = p[3]

    ydata = data_conditioning  # conditioning
    xdata = data_conditioned  # conditioned
    if xdata.ndim < 2:
        xdata = np.expand_dims(xdata, axis=1)
    for vi in range(xdata.shape[1]):
        q += sum(
            np.log(sigma * ydata ** b)
            + 0.5
            * ((xdata[:, vi] - (a * ydata + mu * ydata ** b)) / (sigma * ydata ** b))
            ** 2
        )
    return q
