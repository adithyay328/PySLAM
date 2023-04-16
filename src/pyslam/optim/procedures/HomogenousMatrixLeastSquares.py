# This module contains a simple
# implementation of solving the homogenous
# linear least squares problem for a given
# matrix A using the SVD.

import jax
import jax.numpy as jnp


@jax.jit
@jax.vmap
def homogenousMatrixLeastSquares(
    inMat: jax.Array,
) -> jax.Array:
    # Solve for homography using SVD; basically, pick the row of Vt
    # that corresponds to the smallest singular value. This is,
    # by the construction of the SVD, always the last row
    # of vt

    _, _, vt = jnp.linalg.svd(inMat)

    return vt[-1]
