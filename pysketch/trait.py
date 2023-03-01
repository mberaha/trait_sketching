import numpy as np
import jax.numpy as jnp

from jax import jit, vmap
from jax.scipy.special import gammaln, logsumexp
from functools import partial
from scipy.special import loggamma, softmax

from .utils import log_binom

def poisson_gamma_pred(l, c, b, a, theta, J):
    """
    Computes the posterior distribution of f_{Y_{n+1, l}} for the
    Poisson IBP under the Gamma process prior

    Arguments
    ---------
    l: scalar or array, where to evaluate the posterior mass
    c: the value of C_{h(Y_{n+1, l})}
    b: the value of Z_{n+1}(C_{h(Y_{n+1, l})})
    a: the value of Z_{n+1}(Y_{n+1, l})
    theta: total mass parameter
    J: number of buckets in the sketch
    """
    out = np.log(theta/J) + log_binom(c, l) + log_binom(b, a) + \
        loggamma(l + a) + loggamma(theta/J + c + b - l - a) - loggamma(theta/J + c + b)
    return out


def poisson_gg_pred(l, c, b, a, m, theta, sigma, tau, r, J, log_gen_fac_table):
    """
    Computes the posterior distribution of f_{Y_{n+1, l}} for the
    Poisson IBP under the Generalized Gamma process prior

    Arguments
    ---------
    l: scalar or array, where to evaluate the posterior mass
    c: the value of C_{h(Y_{n+1, l})}
    b: the value of Z_{n+1}(C_{h(Y_{n+1, l})})
    a: the value of Z_{n+1}(Y_{n+1, l})
    theta, sigma, tau: parameters for the generalized gamma process
    r: scale parameter for the Poisson likelihood
    J: number of buckets in the sketch
    log_gen_fac_table: jnp.array, see utils.log_gen_fac
    """
    out = np.zeros(len(l))
    out += np.log(theta/J) + log_binom(c, l) + log_binom(b, a)
    out += np.log(sigma) + loggamma(l + a + 1 - sigma) - loggamma(1 - sigma)
    out+= (sigma - l - a) * np.log((tau + (m+1) * r))
    ub = c - l + b - a
    tmp = np.zeros_like(out)
    for ind in range(len(l)):
        i_range = np.arange(1, ub[ind] + 1)
        tmp[ind] = logsumexp(
            i_range * np.log(theta / J) + 
            log_gen_fac_table[ub[ind], i_range] -
            (ub[ind] - sigma * i_range) * np.log((tau + (m+1) * r))
        )
    out += tmp
    return out


