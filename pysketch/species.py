import cmath
import numpy as np
import jax.numpy as jnp

from jax import vmap
from jax.scipy.special import gammaln, logsumexp
from scipy.special import loggamma, softmax
from mpmath import gammainc, log
from itertools import product

from .utils import log_binom

def py_pred_full(l_max, sketch, gamma, sigma, j, J, log_gen_fac_table):
    """
    Computes the posterior probability of f_{X_{n+1}} on the grid
    `[0, 1, ..., l_max]` under the PYP prior

    Arguments
    ---------
    l_max: size of the grid where to compute the posterior (int)
    sketch: np.array containing the sketched counts
    gamma: total mass parameter (float)
    sigma: discount parameter (float)
    j: bucket into which X_{n+1} is hashed (int)
    J: total number of buckets (with of the sketch, int)
    log_gen_fac_table: jnp.array, see utils.log_gen_fac
    """
    
    def py_pred_aux(i_inds, l):
        s = jnp.sum(i_inds, axis=1)
        out = gammaln((gamma + sigma) / sigma + s) - s * jnp.log(J)
        cks = sketch.at[j].add(-l)
        out += jnp.sum(log_gen_fac_table[cks, i_inds], axis=1)
        return out
    
    
    l_range = jnp.arange(l_max, dtype=int)
    c = sketch[j]
    
    a = jnp.log(gamma / J)
    b = log_binom(c, l_range) + gammaln(l_range + 1 - sigma) - gammaln(1-sigma)
    out = a + b
    
    S_ind_all = jnp.stack(jnp.meshgrid(
        *[jnp.arange(0, sketch[k] + 1) for k in range(J)], indexing="ij")).reshape((J, -1)).T
    S_ind_all = S_ind_all.astype(int)
    A = vmap(lambda l: py_pred_aux(S_ind_all, l))(l_range)
    
    for l in range(l_max):
        mask = S_ind_all[:, j] <= sketch[j] - l
        curr = logsumexp(A[l, mask])
        out = out.at[l].add(curr)
    
    return softmax(out)


def dp_pred(l_max, sketch, gamma, j, J):
    """
    Computes the posterior probability of f_{X_{n+1}} on the grid
    `[0, 1, ..., l_max]` under the DP prior

    Arguments
    ---------
    l_max: size of the grid where to compute the posterior (int)
    sketch: np.array containing the sketched counts
    gamma: total mass parameter (float)
    j: bucket into which X_{n+1} is hashed (int)
    J: total number of buckets (with of the sketch, int)
    """
    out = np.zeros(shape=(l_max+1))
    c = sketch[j]
    for l in range(l_max+1):
        curr = np.log(gamma/J)
        curr += loggamma(c + 1.0) - loggamma(c - l + 1.0)
        curr += loggamma(c - l + gamma/J) - loggamma(c + gamma/J + 1.0)
        out[l] = curr
    return softmax(out)


def py_pred_single(l_max, c, m, theta, alpha, J, log_gen_fac_table):
    """
    Computes the single-bucket posterior probability of f_{X_{n+1}} on the grid
    `[0, 1, ..., l_max]` under the PYP prior. See Dolera, Favaro and Pelucchetti
    (2023) JMLR.

    Arguments
    ---------
    l_max: size of the grid where to compute the posterior (int)
    c: the value of C_{h(X_{n+1})}
    m: total number of data
    theta: total mass parameter (float)
    alpha: discount parameter (float)
    J: total number of buckets (with of the sketch, int)
    log_gen_fac_table: jnp.array, see utils.log_gen_fac
    """
    
    def py_pred_aux(i_inds, l):
        s = jnp.sum(i_inds, axis=1)
        tmp = loggamma((theta + alpha) / alpha + s) - loggamma((theta + alpha) / alpha)
        tmp += i_inds[:, 0] * np.log(1/J) + i_inds[:, 1] * np.log(1 - 1 /J)
        tmp += log_gen_fac_table[c - l, i_inds[:, 0]] + log_gen_fac_table[m - c, i_inds[:, 1]]
        return tmp
    
    l_range = jnp.arange(l_max + 1)
    out = log_binom(c, l_range) + jnp.log(theta / J)
    out += gammaln(l_range + 1 - alpha) - gammaln(1 - alpha)

    S_ind_all = jnp.stack(jnp.meshgrid(
        jnp.arange(c + 1), jnp.arange(m - c + 1), indexing="ij")).reshape((2, -1)).T
    S_ind_all = S_ind_all.astype(int)

    A = vmap(lambda l: py_pred_aux(S_ind_all, l))(l_range)

    for l in range(l_max + 1):
        mask = S_ind_all[:, 0] <= c - l
        curr = logsumexp(A[l, mask])
        out = out.at[l].add(curr)
    
    return softmax(out)


def inc_gamma(a, b):
    return float(log(gammainc(a, b, regularized=False)))


def ngg_pred_single(l_max, c, m, sigma, beta, J, log_gen_fac_table):
    def _aux(i, j):
        tmp = i * np.log(1/J) + j * np.log(1 - 1 /J)
        tmp += log_gen_fac_table[c - l, i] + log_gen_fac_table[m - c, j]
        b = logsumexp(
            [log_binom(m, k) + k * cmath.log(beta) * inc_gamma(i + j - k / sigma, beta ** sigma)
             for k in range(m)]
        )
        return tmp + b.real
    
    out = np.zeros(shape=(l_max+1))
    for l in range(l_max+1):
        curr = np.log(sigma / J)
        curr += log_binom(c, l)
        curr += loggamma(l + 1 - sigma) - loggamma(1 - sigma)
        i_idxs = np.arange(c - l + 1)
        j_idxs = np.arange(m - c + 1)
        ij_idxs = product(i_idxs, j_idxs)

        tmp = [_aux(*ij) for ij in ij_idxs]
        curr += logsumexp(tmp)
        out[l] = curr
    
    print(out)
    return softmax(out)