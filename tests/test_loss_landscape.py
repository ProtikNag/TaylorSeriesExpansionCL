"""Check of the E4 PCA plane: points in a 2-D affine plane are reconstructed exactly."""

import numpy as np

from experiments.loss_landscape import pca_plane


def test_pca_plane_recovers_planar_points() -> None:
    """Six points on a random 2-D plane in R^50: zero residual, explained variance sums to 1."""
    rng = np.random.RandomState(0)
    basis, offset = rng.randn(2, 50), rng.randn(50)
    points = rng.randn(6, 2) @ basis + offset
    mu, directions, coords, explained = pca_plane(points)
    assert np.allclose(directions @ directions.T, np.eye(2))
    assert np.allclose(mu + coords @ directions, points)
    assert np.isclose(explained.sum(), 1.0)
