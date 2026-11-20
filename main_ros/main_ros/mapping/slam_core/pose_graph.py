"""
Pose graph and its least-squares optimizer (graph-based SLAM back-end).

Graph
  * Node i: the pose X_i = (x, y, theta) of keyframe i in the map frame.
  * Edge (i, j): a measurement z_ij of the pose of j seen from i, with an
    information matrix Omega_ij = Sigma_ij^-1 (how much we trust it).
      - "seq"  edges link consecutive keyframes (from the front-end),
      - "loop" edges link a keyframe to an old one it re-observed
        (from loop-closure scan matching).

Error of one edge (Grisetti et al., "A Tutorial on Graph-Based SLAM", 2010):

    e_ij = t2v( Z_ij^-1 · X_i^-1 · X_j )
         = [ R_z^T ( R_i^T (t_j - t_i) - t_z ) ;  wrap(theta_j - theta_i - theta_z) ]

The optimizer finds the node poses that minimize the total weighted error

    F(X) = sum_ij  rho( e_ij^T Omega_ij e_ij )

Linearizing e_ij around the current guess with Jacobians A_ij (w.r.t. X_i) and
B_ij (w.r.t. X_j) gives the sparse normal equations

    H dX = -b,   H = sum J^T Omega J,   b = sum J^T Omega e

solved repeatedly (Gauss-Newton), with Levenberg-Marquardt damping when a step
would increase the error. Node 0 is held fixed: it defines the map frame.

Robustness (bad loop closures must not destroy the map):
  * loop edges use a Huber kernel rho, applied by iteratively re-weighting the
    edge: w = 1 when sqrt(s) <= delta, w = delta / sqrt(s) otherwise;
  * after convergence, a loop edge whose error is still statistically
    impossible (chi^2 above the 99.9% bound for 3 DOF) is declared an outlier,
    removed, and the optimization is redone without it.

H is sparse (each edge touches two nodes), so it is solved with SciPy's sparse
solver. Without SciPy a dense solver is used for small graphs.
"""

from dataclasses import dataclass
import itertools
import math
import time

import numpy as np

from .se2 import wrap_angle

try:
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spla
    HAVE_SCIPY = True
except ImportError:          # pragma: no cover
    HAVE_SCIPY = False

DENSE_MAX_NODES = 1200      # dense fallback limit (memory ~ (3N)^2 * 8 bytes)

_edge_ids = itertools.count()


@dataclass(frozen=True)
class Edge:
    i: int
    j: int
    z: tuple                 # measured pose of j in frame i (x, y, theta)
    info: tuple              # 3x3 information matrix, row-major (9 values)
    kind: str                # "seq" or "loop"
    id: int = -1


def make_edge(i, j, z, info, kind):
    return Edge(int(i), int(j), tuple(float(v) for v in z),
                tuple(float(v) for v in np.asarray(info).ravel()), kind, next(_edge_ids))


class PoseGraph:
    """Container for nodes and edges. Thread safety is handled by the caller."""

    def __init__(self):
        self._poses = np.zeros((256, 3))
        self.n = 0
        self.edges = []

    @property
    def poses(self):
        return self._poses[: self.n]

    def add_node(self, pose):
        if self.n == self._poses.shape[0]:
            self._poses = np.vstack((self._poses, np.zeros_like(self._poses)))
        self._poses[self.n] = pose
        self.n += 1
        return self.n - 1

    def add_edge(self, edge):
        self.edges.append(edge)

    def remove_edges(self, ids):
        ids = set(ids)
        self.edges = [e for e in self.edges if e.id not in ids]

    def loop_edges(self):
        return [e for e in self.edges if e.kind == "loop"]

    def clear(self):
        self.n = 0
        self.edges = []


@dataclass
class OptimizationStats:
    iterations: int = 0
    chi2_before: float = 0.0
    chi2_after: float = 0.0
    rejected_edges: tuple = ()
    max_correction: float = 0.0
    solver: str = ""
    time_s: float = 0.0


# ----------------------------------------------------------------------
# Linearization
# ----------------------------------------------------------------------
def _edge_arrays(edges):
    I = np.array([e.i for e in edges], dtype=np.int64)
    J = np.array([e.j for e in edges], dtype=np.int64)
    Z = np.array([e.z for e in edges], dtype=np.float64).reshape(-1, 3)
    Om = np.array([e.info for e in edges], dtype=np.float64).reshape(-1, 3, 3)
    robust = np.array([e.kind == "loop" for e in edges], dtype=bool)
    return I, J, Z, Om, robust


def _errors(X, I, J, Z):
    ti, tj = X[I, :2], X[J, :2]
    thi, thj = X[I, 2], X[J, 2]
    ci, si = np.cos(thi), np.sin(thi)
    cz, sz = np.cos(Z[:, 2]), np.sin(Z[:, 2])
    d = tj - ti
    rx = ci * d[:, 0] + si * d[:, 1] - Z[:, 0]          # R_i^T d - t_z
    ry = -si * d[:, 0] + ci * d[:, 1] - Z[:, 1]
    e = np.empty((I.size, 3))
    e[:, 0] = cz * rx + sz * ry                          # R_z^T (...)
    e[:, 1] = -sz * rx + cz * ry
    e[:, 2] = wrap_angle(thj - thi - Z[:, 2])
    return e


def _chi2(e, Om):
    return np.einsum("ei,eij,ej->e", e, Om, e)


def _robust_weights(s, robust, delta):
    w = np.ones_like(s)
    r = np.sqrt(np.maximum(s, 0.0))
    big = robust & (r > delta)
    w[big] = delta / r[big]
    return w


def _robust_cost(s, robust, delta):
    r = np.sqrt(np.maximum(s, 0.0))
    cost = s.copy()
    big = robust & (r > delta)
    cost[big] = 2.0 * delta * r[big] - delta ** 2
    return float(cost.sum())


def _jacobians(X, I, J, Z):
    thi = X[I, 2]
    ci, si = np.cos(thi), np.sin(thi)
    cz, sz = np.cos(Z[:, 2]), np.sin(Z[:, 2])
    d = X[J, :2] - X[I, :2]
    a = thi + Z[:, 2]                    # R_z^T R_i^T = R(-a)
    ca, sa = np.cos(a), np.sin(a)
    E = I.size
    A = np.zeros((E, 3, 3))
    B = np.zeros((E, 3, 3))
    A[:, 0, 0], A[:, 0, 1], A[:, 1, 0], A[:, 1, 1] = -ca, -sa, sa, -ca
    B[:, 0, 0], B[:, 0, 1], B[:, 1, 0], B[:, 1, 1] = ca, sa, -sa, ca
    gx = -si * d[:, 0] + ci * d[:, 1]    # (dR_i^T / dtheta_i) d
    gy = -ci * d[:, 0] - si * d[:, 1]
    A[:, 0, 2] = cz * gx + sz * gy
    A[:, 1, 2] = -sz * gx + cz * gy
    A[:, 2, 2] = -1.0
    B[:, 2, 2] = 1.0
    return A, B


def _build_system(X, I, J, Z, Om, w):
    """Return (rows, cols, vals, b) of the normal equations for all 3N unknowns."""
    e = _errors(X, I, J, Z)
    A, B = _jacobians(X, I, J, Z)
    Omw = Om * w[:, None, None]
    AtO = np.einsum("eki,ekj->eij", A, Omw)
    BtO = np.einsum("eki,ekj->eij", B, Omw)
    Hii = AtO @ A
    Hij = AtO @ B
    Hjj = BtO @ B
    bi = np.einsum("eij,ej->ei", AtO, e)
    bj = np.einsum("eij,ej->ei", BtO, e)

    r = np.arange(3)
    def block(ri, ci):
        rows = (3 * ri)[:, None, None] + r[None, :, None]
        cols = (3 * ci)[:, None, None] + r[None, None, :]
        return np.broadcast_to(rows, (ri.size, 3, 3)), np.broadcast_to(cols, (ri.size, 3, 3))

    parts = [(block(I, I), Hii), (block(I, J), Hij),
             (block(J, I), np.transpose(Hij, (0, 2, 1))), (block(J, J), Hjj)]
    rows = np.concatenate([p[0][0].ravel() for p in parts])
    cols = np.concatenate([p[0][1].ravel() for p in parts])
    vals = np.concatenate([p[1].ravel() for p in parts])
    n3 = X.shape[0] * 3
    b = (np.bincount((3 * I[:, None] + r).ravel(), bi.ravel(), minlength=n3)
         + np.bincount((3 * J[:, None] + r).ravel(), bj.ravel(), minlength=n3))
    return rows, cols, vals, b


def _solve(rows, cols, vals, b, n3, lam):
    """Solve (H + lam*diag(H)) dx = -b for nodes 1..N-1 (node 0 fixed)."""
    if HAVE_SCIPY:
        H = _sp.coo_matrix((vals, (rows, cols)), shape=(n3, n3)).tocsc()[3:, 3:]
        if lam > 0:
            H = H + _sp.diags(lam * H.diagonal())
        return _spla.spsolve(H.tocsc(), -b[3:]), "scipy-sparse"
    if n3 // 3 > DENSE_MAX_NODES:
        raise RuntimeError("pose graph too large for the dense solver; install SciPy "
                           "(sudo apt install python3-scipy)")
    H = np.bincount(rows * n3 + cols, vals, minlength=n3 * n3).reshape(n3, n3)[3:, 3:]
    if lam > 0:
        H[np.diag_indices_from(H)] *= (1.0 + lam)
    return np.linalg.solve(H, -b[3:]), "numpy-dense"


# ----------------------------------------------------------------------
# Optimizer
# ----------------------------------------------------------------------
def optimize(poses, edges, max_iterations=30, huber_delta=2.0, tolerance=1e-6,
             outlier_chi2=16.27, max_rejections=5):
    """
    Optimize node poses. Returns (optimized poses, stats).
    The input array is not modified. Loop edges rejected as outliers are
    reported in stats.rejected_edges (edge ids).
    """
    t0 = time.perf_counter()
    X0 = np.array(poses, dtype=np.float64)
    stats = OptimizationStats()
    if X0.shape[0] < 2 or not edges:
        return X0, stats

    active = list(edges)
    rejected = []
    for _ in range(max_rejections + 1):
        X, iters, c0, c1, solver = _optimize_once(X0, active, max_iterations,
                                                  huber_delta, tolerance)
        I, J, Z, Om, robust = _edge_arrays(active)
        s = _chi2(_errors(X, I, J, Z), Om)
        bad = np.flatnonzero(robust & (s > outlier_chi2))
        if bad.size == 0:
            break
        worst = bad[np.argmax(s[bad])]           # remove one at a time, worst first
        rejected.append(active[worst].id)
        active = [e for k, e in enumerate(active) if k != worst]

    stats.iterations = iters
    stats.chi2_before, stats.chi2_after = c0, c1
    stats.rejected_edges = tuple(rejected)
    stats.max_correction = float(np.max(np.hypot(*(X[:, :2] - X0[:, :2]).T)))
    stats.solver = solver
    stats.time_s = time.perf_counter() - t0
    return X, stats


def _optimize_once(X0, edges, max_iterations, delta, tolerance):
    I, J, Z, Om, robust = _edge_arrays(edges)
    X = X0.copy()
    n3 = X.shape[0] * 3

    def cost(Xc):
        s = _chi2(_errors(Xc, I, J, Z), Om)
        return _robust_cost(s, robust, delta), s

    c, s = cost(X)
    c_start = c
    lam, solver, it = 0.0, "", 0
    for it in range(1, max_iterations + 1):
        w = _robust_weights(s, robust, delta)
        rows, cols, vals, b = _build_system(X, I, J, Z, Om, w)
        accepted = False
        for _ in range(10):
            dx, solver = _solve(rows, cols, vals, b, n3, lam)
            if not np.all(np.isfinite(dx)):
                lam = max(lam * 10.0, 1e-4)
                continue
            Xn = X.copy()
            Xn[1:] += dx.reshape(-1, 3)
            Xn[:, 2] = wrap_angle(Xn[:, 2])
            cn, sn = cost(Xn)
            if cn <= c:
                accepted = True
                lam = lam / 10.0 if lam > 1e-7 else 0.0
                break
            lam = max(lam * 10.0, 1e-4)
        if not accepted:
            break
        improvement = (c - cn) / max(c, 1e-12)
        X, c, s = Xn, cn, sn
        if improvement < tolerance or float(np.max(np.abs(dx))) < 1e-6:
            break
    return X, it, c_start, c, solver


def information_from_sigmas(sigma_x, sigma_y, sigma_theta):
    return np.diag([1.0 / sigma_x ** 2, 1.0 / sigma_y ** 2, 1.0 / sigma_theta ** 2])


def information_from_covariance(cov, floor_sigmas):
    """Invert a 3x3 covariance after adding a floor on each standard deviation."""
    cov = np.array(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    cov = cov + np.diag(np.square(floor_sigmas))
    return np.linalg.inv(cov)


def chi2_of_edge(poses, edge):
    I = np.array([edge.i]); J = np.array([edge.j])
    Z = np.array([edge.z]); Om = np.array(edge.info).reshape(1, 3, 3)
    return float(_chi2(_errors(poses, I, J, Z), Om)[0])


def _self_test():                          # pragma: no cover
    """Square loop with a drifting chain: optimization must close it."""
    true = np.array([[0, 0, 0], [1, 0, math.pi / 2], [1, 1, math.pi], [0, 1, -math.pi / 2]], float)
    from .se2 import relative
    edges = []
    for k in range(4):
        z = relative(true[k], true[(k + 1) % 4])
        edges.append(make_edge(k, (k + 1) % 4, z, np.eye(3) * 100,
                               "loop" if k == 3 else "seq"))
    noisy = true + np.array([[0, 0, 0], [0.1, 0.05, 0.05], [0.2, 0.1, 0.1], [0.3, 0.1, 0.15]])
    X, st = optimize(noisy, edges)
    return float(np.abs(X - true).max()), st
