import numpy as np

import jax
import jax.numpy as jnp
from jax import vmap
#from jax.experimental import maps
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.experimental.pjit import pjit

from ott.solvers.linear import sinkhorn
from ott.geometry import pointcloud, geometry, epsilon_scheduler
Epsilon = epsilon_scheduler.Epsilon
PointCloud = pointcloud.PointCloud
Geometry = geometry.Geometry

from ott.geometry import costs

from contextlib import nullcontext

class PNormCost(costs.CostFn):
    def __init__(self, p: float = 2.0):
        self.p = p
    def pairwise(self, x, y):
        # cost(x,y) = ||x-y||_p ** p  (standard p-Wasserstein)
        return jnp.linalg.norm(x - y, ord=self.p) ** self.p


def _pairwise_vmap(func):
    return vmap(vmap(lambda x, y: func(x, y), in_axes=[None, 0]), in_axes=[0, None])


def _get_batch_ranges(data_size, batch_size):
    batch_sizes = [batch_size] * int(np.ceil(data_size / batch_size))
    batch_sizes[-1] -= np.sum(batch_sizes) - data_size
    batch_ranges = np.insert(np.cumsum(batch_sizes), 0, 0)
    batch_ranges = np.stack((batch_ranges[:-1], batch_ranges[1:]), axis=1)
    return batch_ranges


def wasserstein_metric(
    x,
    y,
    p=2,
    cost_fn=None,
    target=1e-1,
    init=10.0,
    decay=0.995,
    threshold=1e-2,
    max_iterations=1000,
    geometry_type="pointcloud",
    batch_size=None,
    **sinkhorn_kwargs,
):
    # choose a cost function
    # - p==2 and cost_fn is None: use PointCloud default (squared Euclidean)
    # - otherwise: use provided cost_fn or a generic p-norm cost (||·||_p^p)
    use_cost_fn = cost_fn if cost_fn is not None else (None if p == 2 else PNormCost(p=p))

    if geometry_type == "pointcloud":
        geom = PointCloud(
            x,
            y,
            cost_fn=use_cost_fn,                # None -> default squared Euclidean
            epsilon=Epsilon(target=target,      # pass ε directly
                            init=init,
                            decay=decay),
        )
        sol = sinkhorn(
            geom,
            threshold=threshold,
            max_iterations=max_iterations,
            **sinkhorn_kwargs,
        )

    elif geometry_type == "precompute":
        # Build a cost matrix equal to distance^p
        if batch_size is None:
            M = (
                cost_fn.norm(x)[:, None]
                + cost_fn.norm(y)[None, :]
                + _pairwise_vmap(cost_fn.pairwise)(x, y)
            ) ** (0.5 * p)   # (sqrt(·))^p = distance^p
        else:
            if not isinstance(batch_size, tuple):
                batch_size = (batch_size, batch_size)
            batch_ranges = [
                _get_batch_ranges(x.shape[0], batch_size[0]),
                _get_batch_ranges(y.shape[0], batch_size[1]),
            ]
            M_blocks = []
            for i_range in batch_ranges[0]:
                row_blocks = []
                for j_range in batch_ranges[1]:
                    blk = _pairwise_vmap(cost_fn.pairwise)(
                        x[i_range[0] : i_range[1]],
                        y[j_range[0] : j_range[1]],
                    )
                    row_blocks.append(blk)
                M_blocks.append(jnp.block(row_blocks))
            M = jnp.block(M_blocks)
            M = (cost_fn.norm(x)[:, None] + cost_fn.norm(y)[None, :] + M) ** (0.5 * p)

        geom = Geometry(
            cost_matrix=M,
            epsilon=Epsilon(target=target, init=init, decay=decay),  # no power
        )
        sol = sinkhorn(
            geom,
            threshold=threshold,
            max_iterations=max_iterations,
            **sinkhorn_kwargs,
        )

    return sol.reg_ot_cost, sol.converged, 10 * jnp.sum(sol.errors > -1)



def distance_matrix(
    data, metric=wasserstein_metric, p=2, batch_size=None, mesh_shape=None
):
    if mesh_shape is not None:
        devices = np.asarray(jax.devices()).reshape(*mesh_shape)
        mesh = Mesh(devices, ("x", "y"))

        pairwise_cost = pjit(
            _pairwise_vmap(metric),
            in_axis_resources=(
                P("x", None, None),
                P("y", None, None),
            ),
            out_axis_resources=P("x", "y"),
        )

    else:
        mesh = nullcontext()
        if batch_size is None:
            data = jnp.asarray(data)

        def pairwise_cost(x, y):
            cost_mat, converged, steps = _pairwise_vmap(metric)(
                jnp.asarray(x), jnp.asarray(y)
            )
            return np.asarray(cost_mat), np.asarray(converged), np.asarray(steps)

    with mesh:
        if batch_size is None:
            cost_mat, converged, steps = pairwise_cost(data, data)
        else:
            if not isinstance(batch_size, tuple):
                batch_size = (batch_size, batch_size)
            batch_ranges = [
                _get_batch_ranges(data.shape[0], batch_size[0]),
                _get_batch_ranges(data.shape[0], batch_size[1]),
            ]

            cost_mat = np.empty((data.shape[0], data.shape[0]))
            converged = np.empty((data.shape[0], data.shape[0]), dtype=bool)
            steps = np.empty((data.shape[0], data.shape[0]), dtype=int)
            for i, i_range in enumerate(batch_ranges[0]):
                for j, j_range in enumerate(batch_ranges[1]):
                    print(
                        f"Distance matrix: batch {i*batch_ranges[1].shape[0] + j + 1}"
                        f" of {batch_ranges[0].shape[0]*batch_ranges[1].shape[0]}"
                    )
                    out = pairwise_cost(
                        data[i_range[0] : i_range[1]], data[j_range[0] : j_range[1]]
                    )
                    cost_mat[i_range[0] : i_range[1], j_range[0] : j_range[1]] = out[0]
                    converged[i_range[0] : i_range[1], j_range[0] : j_range[1]] = out[1]
                    steps[i_range[0] : i_range[1], j_range[0] : j_range[1]] = out[2]

    # symmetrize & debias, https://arxiv.org/pdf/2006.02575.pdf
    cost_diag = np.diag(cost_mat)
    dist_mat = ((cost_mat + cost_mat.T - cost_diag - cost_diag[:, None]) / 2) ** (1 / p)

    return dist_mat, converged, steps
