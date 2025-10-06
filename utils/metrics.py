import numpy as np

import jax
import jax.numpy as jnp
from jax import vmap
#from jax.experimental import maps
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.experimental.pjit import pjit

#from ott.solvers.linear import sinkhorn
from ott.solvers.linear.sinkhorn import Sinkhorn
from ott.problems.linear import linear_problem as lp
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
        # cost(x,y) = ||x-y||_p ** p  (works for general p>=1)
        return jnp.sum(jnp.abs(x - y) ** self.p)


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
        #
        solver = Sinkhorn(
            threshold=threshold,
            max_iterations=max_iterations,
            **sinkhorn_kwargs,
        )
        prob = lp.LinearProblem(geom)           # <- wrap Geometry
        sol = solver(prob)

    #->
    elif geometry_type == "precompute":
        # Build cost matrix M = ||x - y||^p using a pairwise function
        if cost_fn is None and p == 2:
            def sqeuclid(a, b):  # squared L2
                return jnp.sum((a - b) ** 2, axis=-1)
            pair = _pairwise_vmap(sqeuclid)
        else:
            cf = use_cost_fn if use_cost_fn is not None else PNormCost(p=p)
            pair = _pairwise_vmap(cf.pairwise)  # returns ||·||_p^p

        if batch_size is None:
            M = pair(x, y)
        else:
            if not isinstance(batch_size, tuple):
                batch_size = (batch_size, batch_size)
            xr = _get_batch_ranges(x.shape[0], batch_size[0])
            yr = _get_batch_ranges(y.shape[0], batch_size[1])
            rows = []
            for i0, i1 in xr:
                row_blocks = []
                for j0, j1 in yr:
                    row_blocks.append(pair(x[i0:i1], y[j0:j1]))
                rows.append(jnp.block(row_blocks))
            M = jnp.block(rows)

        geom = Geometry(cost_matrix=M,
                        epsilon=Epsilon(target=target, init=init, decay=decay))
        solver = Sinkhorn(threshold=threshold, max_iterations=max_iterations, **sinkhorn_kwargs)
        prob = lp.LinearProblem(geom)
        sol = solver(prob)
    ###-#

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
    dist_mat = ((cost_mat + cost_mat.T - cost_diag - cost_diag[:, None]) / 2)
    assert dist_mat.min() < 1e-2, 'TOO NEGATIVE!'
    dist_mat = np.clip(dist_mat, 0., None)  # enforce positivity, avoid spurious small neg
    dist_mat = dist_mat ** (1 / p)

    return dist_mat, converged, steps
