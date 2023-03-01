import numpy as np

import jax.numpy as jnp
from jax import jit
from jax.scipy.special import gammaln


def log_binom(a, b):
    """
    Computes the log of the binomial coefficient.
    If one argument is a scalar and the other is a vector, the 
    scalar one gets promoted.
    """
    a = jnp.atleast_1d(a)
    b = jnp.atleast_1d(b)
    
    if len(a) > 1 and len(b) == 1:
        b = jnp.repeat(b, len(a))
    elif len(b) > 1 and len(a) == 1:
        a = jnp.repeat(a, len(b))
        
    out = gammaln(a+1) - gammaln(b+1) - gammaln(a - b + 1)
    
    w1 = jnp.where((b == 0) | (a == b))[0]
    out = out.at[w1].set(jnp.log(1))
    
    w2 = jnp.where((a == 0) | (a < b))[0]
    out = out.at[w2].set(-jnp.inf)
    
    return out


def log_gen_fac(sigma, i_max):
    """
    Computes a table of the (log) of generalized factorial 
    coefficients until the index `i_max`.
    Optimized for numerical precision, return a numpy matrix
    """
    out = np.zeros(shape=(i_max+1, i_max+1)) + (-np.infty)
    out[0, 0] = 0
    out[1, 1] = np.log(sigma)
    for i in range(2, i_max+1):
        out[i, 1] = np.log(-(sigma-(i-1))) + out[i-1, 1]
        out[i, i] = np.log(sigma) + out[i-1, i-1]
        
        for j in range(2, i):
            a = np.log(sigma) + out[i-1, j-1];
            b = np.log((i-1) - (sigma * j)) + out[i-1, j];
            out[i,j] = max(a,b) + np.log(1 + np.exp(min(a,b)- max(a,b)))
    return out